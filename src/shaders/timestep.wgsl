// One time step of plate motion, as six compute passes over the world
// cells (and, for aging, over every plate's samples). Mirrors
// `Frames::step` in timestep.rs, which documents the rules and is what the
// tests check; keep the two in step.
//
// Prepended: generated constants (timestep::wgsl_prelude), cube.wgsl and
// lookup.wgsl.

// Layout shared with timestep.rs.
struct Sample {
    flags: u32,
    crust_km: f32,
    felsic: f32,
    mantle_km: f32,
    depletion: f32,
    age_myr: f32,
    sediment_km: f32,
    relief_m: f32,
};

// Fixed per world cell: its center, and the mantle-fixed relief there.
struct WorldCell {
    center: vec3<f32>,
    dynamic_m: f32,
    swell_m: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

// What the renderer reads: must match CellData in renderer.rs.
struct CellData {
    plate: u32,
    elevation_m: f32,
    age_myr: f32,
    thickness_km: f32,
    dynamic_m: f32,
    felsic: f32,
    lithosphere_km: f32,
    density_excess: f32,
    rock: u32,
    province: u32,
};

// What the CPU reads back: must match ColumnOut in timestep.rs.
struct ColumnOut {
    plate: u32,
    flags: u32,
    sediment_km: f32,
    crust_km: f32,
    felsic: f32,
    mantle_km: f32,
    depletion: f32,
    age_myr: f32,
    elevation_m: f32,
    _pad: f32,
};

struct Params {
    cells: u32,
    plates: u32,
    dt_myr: f32,
    _pad: f32,
    // Rotation from each plate's frame to the globe (quaternion xyzw).
    rotation: array<vec4<f32>, MAX_PLATES>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> world: array<WorldCell>;
@group(0) @binding(2) var<storage, read> neighbors: array<u32>;
@group(0) @binding(3) var<storage, read_write> samples: array<Sample>;
// [0, plates * cells): whether each sample exists (0 or 1).
// [plates * cells, 2 * plates * cells): crust merged into it this step (m).
// [2, 3) * plates * cells: crust it spills to each living neighbor (km,
// f32 bits), then [3, 4): that crust's felsic share (f32 bits).
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>>;
// Per world cell: winning cover, two losers, and new crust (packed, NONE
// when empty).
@group(0) @binding(5) var<storage, read_write> scratch: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read_write> cells: array<CellData>;
@group(0) @binding(7) var<storage, read_write> columns: array<ColumnOut>;
@group(0) @binding(8) var cell_ids: texture_2d_array<u32>;

const NONE: u32 = 0xffffffffu;
const MERGE: u32 = 0x80000000u;

// A plate and its reference cell in one u32: plate in bits 24-30, cell in
// the low 24, bit 31 the merge flag.
fn pack(plate: u32, cell: u32) -> u32 {
    return (plate << 24u) | cell;
}

fn plate_of(x: u32) -> u32 {
    return (x >> 24u) & 0x7fu;
}

fn index_of(x: u32) -> u32 {
    return plate_of(x) * params.cells + (x & 0xffffffu);
}

fn total() -> u32 {
    return params.plates * params.cells;
}

// Sample `i`'s k-th neighbor on the same plate, or NONE.
fn neighbor(i: u32, k: u32) -> u32 {
    let p = i / params.cells;
    let nb = neighbors[(i % params.cells) * 6u + k];
    if nb == NONE {
        return NONE;
    }
    return p * params.cells + nb;
}

// Rotates v by quaternion q, or by its inverse.
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn unrotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return rotate(vec4<f32>(-q.xyz, q.w), v);
}

fn alive(i: u32) -> bool {
    return atomicLoad(&counters[i]) != 0u;
}

fn is_continental(s: Sample) -> bool {
    return (s.flags & FLAG_CONTINENTAL) != 0u;
}

// Column physics, as in crust.rs.

fn ocean_depth_m(age_myr: f32) -> f32 {
    let age = max(age_myr, 0.0);
    if age < 20.0 {
        return -(2500.0 + 350.0 * sqrt(age));
    }
    return -(6400.0 - 3200.0 * exp(-age / 62.8));
}

fn oceanic_mantle_km(age_myr: f32) -> f32 {
    let subsidence_km = (ocean_depth_m(0.0) - ocean_depth_m(age_myr)) / 1000.0;
    return subsidence_km * (RHO_ASTHENOSPHERE - RHO_WATER) / THERMAL_EXCESS;
}

fn mantle_km(s: Sample) -> f32 {
    if is_continental(s) {
        return s.mantle_km;
    }
    return oceanic_mantle_km(s.age_myr);
}

fn crust_density(felsic: f32) -> f32 {
    return RHO_MAFIC_CRUST + (RHO_FELSIC_CRUST - RHO_MAFIC_CRUST) * clamp(felsic, 0.0, 1.0);
}

fn mantle_density(depletion: f32) -> f32 {
    return RHO_ASTHENOSPHERE + THERMAL_EXCESS - DEPLETION_DEFICIT * clamp(depletion, 0.0, 1.0);
}

fn buoyancy(s: Sample) -> f32 {
    return (RHO_ASTHENOSPHERE - RHO_SEDIMENT) * s.sediment_km
        + (RHO_ASTHENOSPHERE - crust_density(s.felsic)) * s.crust_km
        + (RHO_ASTHENOSPHERE - mantle_density(s.depletion)) * mantle_km(s);
}

fn column_elevation_m(s: Sample) -> f32 {
    let b = buoyancy(s) - SEA_LEVEL_BUOYANCY;
    let support = select(RHO_ASTHENOSPHERE - RHO_WATER, RHO_ASTHENOSPHERE, b >= 0.0);
    return 1000.0 * b / support;
}

fn new_ocean() -> Sample {
    return Sample(
        (PROVINCE_OCEANIC << PROVINCE_SHIFT) | (ROCK_BASALT << ROCK_SHIFT),
        OCEANIC_THICKNESS_KM, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    );
}

// Surface rock as it shows (Sample::rock).
fn rock(s: Sample) -> u32 {
    let stored = (s.flags >> ROCK_SHIFT) & 7u;
    if !is_continental(s) && (stored == ROCK_BASALT || stored == ROCK_OCEAN_SEDIMENT) {
        return select(ROCK_OCEAN_SEDIMENT, ROCK_BASALT, s.sediment_km < BARE_SEAFLOOR_KM);
    }
    return stored;
}

// 1. Each world cell finds the crust covering it; the most buoyant wins.
@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = id.x;
    if d >= params.cells {
        return;
    }
    let center = world[d].center;
    var cover = array<u32, MAX_CANDIDATES>();
    var lift = array<f32, MAX_CANDIDATES>();
    var continental = array<bool, MAX_CANDIDATES>();
    var count = 0u;
    for (var p = 0u; p < params.plates; p++) {
        let cell = cell_toward(normalize(unrotate(params.rotation[p], center)));
        if cell == NO_CELL || count >= MAX_CANDIDATES {
            continue;
        }
        let i = p * params.cells + cell;
        if alive(i) {
            cover[count] = pack(p, cell);
            lift[count] = buoyancy(samples[i]);
            continental[count] = is_continental(samples[i]);
            count += 1u;
        }
    }
    if count == 0u {
        scratch[d] = vec4<u32>(NONE);
        return;
    }
    // Continental crust first, then the most buoyant; ties go to the
    // lowest plate, which came first.
    var winner = 0u;
    for (var k = 1u; k < count; k++) {
        let w = winner;
        if (continental[k] && !continental[w])
            || (continental[k] == continental[w] && lift[k] > lift[w]) {
            winner = k;
        }
    }
    var losers = vec2<u32>(NONE);
    var lost = 0u;
    for (var k = 0u; k < count; k++) {
        if k == winner {
            continue;
        }
        let buoyant = continental[k] || min(lift[winner], lift[k]) >= COLLISION_BUOYANCY;
        let merge = select(0u, MERGE, buoyant);
        losers[lost] = cover[k] | merge;
        lost += 1u;
    }
    scratch[d] = vec4<u32>(cover[winner], losers.x, losers.y, NONE);
}

// 2. Cells nothing covers grow new crust on the most common neighboring
// plate (ties to the lowest id).
@compute @workgroup_size(64)
fn diverge(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = id.x;
    if d >= params.cells || scratch[d].x != NONE {
        return;
    }
    var counts = array<u32, MAX_PLATES>();
    for (var k = 0u; k < 6u; k++) {
        let nb = neighbors[d * 6u + k];
        if nb != NONE {
            let o = scratch[nb].x;
            if o != NONE {
                counts[plate_of(o)] += 1u;
            }
        }
    }
    var best = 0u;
    for (var q = 1u; q < params.plates; q++) {
        if counts[q] > counts[best] {
            best = q;
        }
    }
    if counts[best] > 0u {
        let cell = cell_toward(normalize(unrotate(params.rotation[best], world[d].center)));
        if cell != NO_CELL {
            scratch[d].w = pack(best, cell);
        }
    }
}

// 3. Losers are removed (once each; continental and other buoyant ones
// merge into the winner),
// and new crust is planted.
@compute @workgroup_size(64)
fn apply(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = id.x;
    if d >= params.cells {
        return;
    }
    let s = scratch[d];
    for (var k = 0u; k < 2u; k++) {
        let loser = select(s.y, s.z, k == 1u);
        if loser == NONE {
            continue;
        }
        let i = index_of(loser);
        let was = atomicExchange(&counters[i], 0u);
        if was != 0u && (loser & MERGE) != 0u && s.x != NONE {
            let meters = u32(samples[i].crust_km * 1000.0);
            atomicAdd(&counters[total() + index_of(s.x)], meters);
        }
    }
    if s.w != NONE {
        let i = index_of(s.w);
        if !alive(i) {
            samples[i] = new_ocean();
            atomicStore(&counters[i], 1u);
        }
    }
}

// 4. Every sample ages: seafloor gets older and gathers sediment; merged
// crust thickens its winner, and crust past the cap is set aside to spill
// to the sample's living neighbors.
@compute @workgroup_size(64)
fn age(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= total() {
        return;
    }
    let added = f32(atomicExchange(&counters[total() + i], 0u)) / 1000.0;
    atomicStore(&counters[2u * total() + i], 0u);
    if !alive(i) {
        return;
    }
    var s = samples[i];
    let crust = s.crust_km + added;
    var living = 0u;
    for (var k = 0u; k < 6u; k++) {
        let j = neighbor(i, k);
        if j != NONE && alive(j) {
            living += 1u;
        }
    }
    s.crust_km = crust;
    if crust > MAX_CRUST_KM && living > 0u {
        let share = (crust - MAX_CRUST_KM) / f32(living);
        atomicStore(&counters[2u * total() + i], bitcast<u32>(share));
        atomicStore(&counters[3u * total() + i], bitcast<u32>(s.felsic));
        s.crust_km = MAX_CRUST_KM;
    }
    if !is_continental(s) {
        s.age_myr += params.dt_myr;
        s.sediment_km += PELAGIC_KM_PER_MYR * params.dt_myr;
    }
    samples[i] = s;
}

// 5. Crust set aside spills into the neighbors; seafloor buried deep
// enough under it becomes continental (timestep::spill).
@compute @workgroup_size(64)
fn spill(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= total() || !alive(i) {
        return;
    }
    var km = 0.0;
    var felsic_km = 0.0;
    for (var k = 0u; k < 6u; k++) {
        let j = neighbor(i, k);
        if j != NONE {
            let share = bitcast<f32>(atomicLoad(&counters[2u * total() + j]));
            km += share;
            felsic_km += share * bitcast<f32>(atomicLoad(&counters[3u * total() + j]));
        }
    }
    if km <= 0.0 {
        return;
    }
    var s = samples[i];
    s.felsic = (s.felsic * s.crust_km + felsic_km) / (s.crust_km + km);
    s.crust_km += km;
    if !is_continental(s) && s.crust_km >= ACCRETED_KM {
        s.mantle_km = oceanic_mantle_km(s.age_myr);
        s.depletion = 0.0;
        s.flags = FLAG_CONTINENTAL | (PROVINCE_PHANEROZOIC << PROVINCE_SHIFT)
            | (ROCK_SEDIMENT << ROCK_SHIFT);
    }
    samples[i] = s;
}

// 6. Each world cell shows its crust: column, isostatic elevation, and the
// values the renderer and the CPU read.
@compute @workgroup_size(64)
fn view(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = id.x;
    if d >= params.cells {
        return;
    }
    var cover = scratch[d].x;
    if cover == NONE {
        cover = scratch[d].w;
    }
    if cover == NONE {
        // Still a gap: keep showing what was there.
        columns[d].plate = NONE;
        return;
    }
    let s = samples[index_of(cover)];
    let here = world[d];
    let mantle = mantle_km(s);
    let lithosphere = s.sediment_km + s.crust_km + mantle;
    let elevation = column_elevation_m(s) + s.relief_m + here.dynamic_m + here.swell_m;
    let plate = plate_of(cover);
    cells[d] = CellData(
        plate,
        elevation,
        select(s.age_myr, -1.0, is_continental(s)),
        s.crust_km,
        here.dynamic_m,
        s.felsic,
        lithosphere,
        -buoyancy(s) / lithosphere,
        rock(s),
        (s.flags >> PROVINCE_SHIFT) & 3u,
    );
    columns[d] = ColumnOut(
        plate,
        (s.flags & ~(7u << ROCK_SHIFT)) | (rock(s) << ROCK_SHIFT),
        s.sediment_km,
        s.crust_km,
        s.felsic,
        mantle,
        s.depletion,
        s.age_myr,
        elevation,
        0.0,
    );
}
