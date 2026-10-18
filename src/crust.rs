//! Crust and lithosphere: what each grid cell is made of, and how high it
//! sits.
//!
//! Crust type belongs to cells, not plates: like Earth's North American
//! plate, one plate can carry a continent and half an ocean.
//!
//! Each cell is a column of sediment, crust and cold mantle lithosphere
//! floating on the hot asthenosphere, and its elevation follows from the
//! column's buoyancy (isostasy). Densities follow from composition:
//!
//! * Crust runs from mafic (basalt and gabbro, about 2,900 kg/m³) to felsic
//!   (granite, about 2,700 kg/m³); oceanic crust is mafic, bulk continental
//!   crust roughly andesitic, halfway (Rudnick & Gao 2003).
//! * Mantle lithosphere is denser than the asthenosphere because it is
//!   cold, less so where melting long ago depleted it of iron (cratons; the
//!   "isopycnic" idea of Jordan 1975).
//! * Oceanic lithosphere thickens with age as it cools, which makes old
//!   seafloor sink: the depth-age curve of Parsons & Sclater (1977) sets
//!   that thickening, giving about 40 km at 20 Myr and 80 km at 100 Myr,
//!   levelling off in the 90s like seismic estimates for the old Pacific.

use crate::grid::Grid;
use crate::plates::{flood_fill, smooth_noise};
use crate::rng::Pcg32;

/// Share of the surface that is continental crust (Earth: about 40%,
/// counting continental shelves).
pub const CONTINENTAL_FRACTION: f64 = 0.35;
const CONTINENTS: usize = 6;
/// Continent seeds closer than this (radians) are rejected.
const CONTINENT_SPACING: f64 = 0.5;
/// Coastline roughness: crossing cost is exp(COAST_ROUGHNESS * noise).
const COAST_ROUGHNESS: f64 = 1.5;

pub const OCEANIC_THICKNESS_KM: f64 = 7.0;

/// Densities (kg/m³).
pub const RHO_ASTHENOSPHERE: f64 = 3300.0;
const RHO_WATER: f64 = 1030.0;
pub const RHO_SEDIMENT: f64 = 2200.0;
const RHO_MAFIC_CRUST: f64 = 2900.0;
const RHO_FELSIC_CRUST: f64 = 2700.0;
/// Mantle lithosphere's excess density over the asthenosphere from being
/// cold, averaged through its thickness, and how much full depletion takes
/// back (Archean roots run 30 to 80 kg/m³ lighter than primitive mantle).
const THERMAL_EXCESS: f64 = 90.0;
const DEPLETION_DEFICIT: f64 = 90.0;
/// New seafloor sits this deep (m): sets sea level for the isostasy.
const RIDGE_DEPTH_M: f64 = 2500.0;
/// Buoyancy (see [`Column::buoyancy`]) of a column whose top is at sea
/// level: that of new seafloor, plus what it takes to lift it from the
/// ridge depth.
const SEA_LEVEL_BUOYANCY: f64 = (RHO_ASTHENOSPHERE - RHO_MAFIC_CRUST) * OCEANIC_THICKNESS_KM
    + RIDGE_DEPTH_M / 1000.0 * (RHO_ASTHENOSPHERE - RHO_WATER);

/// The physics constants above as WGSL declarations, so the GPU time step
/// computes columns and isostasy with exactly these values.
pub fn wgsl_constants() -> String {
    [
        ("OCEANIC_THICKNESS_KM", OCEANIC_THICKNESS_KM),
        ("RHO_ASTHENOSPHERE", RHO_ASTHENOSPHERE),
        ("RHO_WATER", RHO_WATER),
        ("RHO_SEDIMENT", RHO_SEDIMENT),
        ("RHO_MAFIC_CRUST", RHO_MAFIC_CRUST),
        ("RHO_FELSIC_CRUST", RHO_FELSIC_CRUST),
        ("THERMAL_EXCESS", THERMAL_EXCESS),
        ("DEPLETION_DEFICIT", DEPLETION_DEFICIT),
        ("RIDGE_DEPTH_M", RIDGE_DEPTH_M),
        ("SEA_LEVEL_BUOYANCY", SEA_LEVEL_BUOYANCY),
    ]
    .iter()
    .map(|(name, v)| format!("const {name}: f32 = {v:?};\n"))
    .collect()
}

/// Crust density (kg/m³) for a felsic fraction from 0 (mafic) to 1.
pub fn crust_density(felsic: f64) -> f64 {
    RHO_MAFIC_CRUST + (RHO_FELSIC_CRUST - RHO_MAFIC_CRUST) * felsic.clamp(0.0, 1.0)
}

/// Mantle lithosphere density (kg/m³) for a depletion from 0 (fertile) to
/// 1 (strongly depleted).
pub fn mantle_lithosphere_density(depletion: f64) -> f64 {
    RHO_ASTHENOSPHERE + THERMAL_EXCESS - DEPLETION_DEFICIT * depletion.clamp(0.0, 1.0)
}

/// The felsic fraction that gives `crust_km` of crust a buoyancy (see
/// [`Column::buoyancy`]) of `buoyancy`.
pub fn felsic_for(crust_km: f64, buoyancy: f64) -> f64 {
    if crust_km <= 0.0 {
        return 0.0;
    }
    let density = RHO_ASTHENOSPHERE - buoyancy / crust_km;
    ((RHO_MAFIC_CRUST - density) / (RHO_MAFIC_CRUST - RHO_FELSIC_CRUST)).clamp(0.0, 1.0)
}

/// The depletion that gives `mantle_km` of mantle lithosphere a buoyancy
/// (see [`Column::buoyancy`]) of `buoyancy`.
pub fn depletion_for(mantle_km: f64, buoyancy: f64) -> f64 {
    if mantle_km <= 0.0 {
        return 0.0;
    }
    ((buoyancy / mantle_km + THERMAL_EXCESS) / DEPLETION_DEFICIT).clamp(0.0, 1.0)
}

/// One cell's lithosphere, top to bottom. Thicknesses in km.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Column {
    pub sediment_km: f64,
    pub crust_km: f64,
    /// Crust composition: 0 mafic (basaltic) to 1 felsic (granitic).
    pub felsic: f64,
    pub mantle_km: f64,
    /// Mantle lithosphere depletion: 0 fertile to 1 strongly depleted.
    pub depletion: f64,
}

impl Column {
    /// Oceanic lithosphere `age` Myr old, with `sediment_km` on top.
    pub fn oceanic(age: f64, sediment_km: f64) -> Self {
        Self {
            sediment_km,
            crust_km: OCEANIC_THICKNESS_KM,
            felsic: 0.0,
            mantle_km: oceanic_mantle_km(age),
            depletion: 0.0,
        }
    }

    /// Total lithosphere thickness (km).
    pub fn thickness_km(&self) -> f64 {
        self.sediment_km + self.crust_km + self.mantle_km
    }

    /// Mass deficit against the asthenosphere: the sum over layers of
    /// (ρ_asthenosphere − ρ_layer) × thickness, in kg/m³·km. Higher floats
    /// higher.
    pub fn buoyancy(&self) -> f64 {
        (RHO_ASTHENOSPHERE - RHO_SEDIMENT) * self.sediment_km
            + (RHO_ASTHENOSPHERE - crust_density(self.felsic)) * self.crust_km
            + (RHO_ASTHENOSPHERE - mantle_lithosphere_density(self.depletion)) * self.mantle_km
    }

    /// Surface elevation (m) by isostasy: the column floats until its
    /// weight matches the asthenosphere it displaces. Below sea level the
    /// water on top counts too.
    pub fn elevation_m(&self) -> f64 {
        let b = self.buoyancy() - SEA_LEVEL_BUOYANCY;
        let support = if b >= 0.0 {
            RHO_ASTHENOSPHERE
        } else {
            RHO_ASTHENOSPHERE - RHO_WATER
        };
        1000.0 * b / support
    }

    /// Mean density minus the asthenosphere's (kg/m³): positive means the
    /// column is heavier than the mantle under it and would sink if pushed.
    pub fn density_excess(&self) -> f64 {
        -self.buoyancy() / self.thickness_km()
    }
}

/// Oceanic mantle lithosphere thickness (km) at `age` Myr: the cold, dense
/// layer whose added weight sinks the seafloor along the depth-age curve.
pub fn oceanic_mantle_km(age: f64) -> f64 {
    let subsidence_km = (ocean_depth_m(0.0) - ocean_depth_m(age)) / 1000.0;
    subsidence_km * (RHO_ASTHENOSPHERE - RHO_WATER) / THERMAL_EXCESS
}

/// What a cell's surface is made of, for display; derived from where it
/// sits and what has happened there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceRock {
    OceanSediment,
    Basalt,
    Andesite,
    Granite,
    Metamorphic,
    Sediment,
    FloodBasalt,
}

impl SurfaceRock {
    pub const ALL: [SurfaceRock; 7] = [
        SurfaceRock::OceanSediment,
        SurfaceRock::Basalt,
        SurfaceRock::Andesite,
        SurfaceRock::Granite,
        SurfaceRock::Metamorphic,
        SurfaceRock::Sediment,
        SurfaceRock::FloodBasalt,
    ];

    pub fn name(self) -> &'static str {
        match self {
            SurfaceRock::OceanSediment => "Deep-sea sediment",
            SurfaceRock::Basalt => "Basalt",
            SurfaceRock::Andesite => "Andesite",
            SurfaceRock::Granite => "Granite and gneiss",
            SurfaceRock::Metamorphic => "Metamorphic rock",
            SurfaceRock::Sediment => "Sedimentary cover",
            SurfaceRock::FloodBasalt => "Flood basalt",
        }
    }

    /// Where it occurs, briefly.
    pub fn origin(self) -> &'static str {
        match self {
            SurfaceRock::OceanSediment => "clays and oozes on older seafloor",
            SurfaceRock::Basalt => "young seafloor and hotspot volcanoes",
            SurfaceRock::Andesite => "volcanic arcs above subduction zones",
            SurfaceRock::Granite => "uplifted, eroded continental basement",
            SurfaceRock::Metamorphic => "collision mountain belts",
            SurfaceRock::Sediment => "continental lowlands, shelves and rift basins",
            SurfaceRock::FloodBasalt => "hotspot volcanism on continents",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrustKind {
    Continental,
    Oceanic,
}

impl CrustKind {
    pub fn is_continental(self) -> bool {
        self == CrustKind::Continental
    }
}

/// Continental crust as a few blobs with rough coasts, covering
/// [`CONTINENTAL_FRACTION`] of the cells. Each continent grows from a seed
/// at its own speed, and all stop at the same moment, so sizes vary.
///
/// Growth order sets crustal age: continents grew by adding younger belts
/// around ancient cores (North America's Canadian Shield ringed by younger
/// mountain belts), so the first cells claimed become Archean cratons, the
/// next Proterozoic, the last Phanerozoic. Returns each cell's kind and
/// province.
pub fn continents(grid: &Grid, rng: &mut Pcg32) -> (Vec<CrustKind>, Vec<CrustProvince>) {
    let mut seeds: Vec<u32> = Vec::new();
    while seeds.len() < CONTINENTS {
        let cell = (rng.next_f64() * grid.len() as f64) as u32;
        let p = grid.centers[cell as usize];
        if seeds
            .iter()
            .all(|&s| grid.centers[s as usize].angle_between(p) >= CONTINENT_SPACING)
        {
            seeds.push(cell);
        }
    }
    let speeds: Vec<f64> = seeds.iter().map(|_| rng.range(0.5, 1.5)).collect();
    let coarse = smooth_noise(grid, rng, 30);
    let fine = smooth_noise(grid, rng, 4);
    let cost: Vec<f64> = coarse
        .iter()
        .zip(&fine)
        .map(|(c, f)| (COAST_ROUGHNESS * (0.7 * c + 0.3 * f)).exp())
        .collect();
    let (_, arrival) = flood_fill(grid, &seeds, &speeds, &cost);

    let mut order: Vec<usize> = (0..grid.len()).collect();
    order.sort_by(|&a, &b| arrival[a].total_cmp(&arrival[b]));
    let mut kind = vec![CrustKind::Oceanic; grid.len()];
    let mut province = vec![CrustProvince::Oceanic; grid.len()];
    let land = (CONTINENTAL_FRACTION * grid.len() as f64) as usize;
    let archean = (ARCHEAN_SHARE * land as f64) as usize;
    let proterozoic = archean + (PROTEROZOIC_SHARE * land as f64) as usize;
    for (rank, &cell) in order[..land].iter().enumerate() {
        kind[cell] = CrustKind::Continental;
        province[cell] = if rank < archean {
            CrustProvince::Archean
        } else if rank < proterozoic {
            CrustProvince::Proterozoic
        } else {
            CrustProvince::Phanerozoic
        };
    }
    (kind, province)
}

/// Shares of continental area by age province, oldest first (roughly
/// Earth's; the rest is Phanerozoic).
const ARCHEAN_SHARE: f64 = 0.15;
const PROTEROZOIC_SHARE: f64 = 0.45;

/// Continental crust by age of formation, with the lithosphere each tends
/// to have (Artemieva & Mooney 2001: over 200 km under Archean cratons,
/// 140 to 200 km under Proterozoic, 100 to 150 km under Phanerozoic belts).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrustProvince {
    Oceanic,
    /// Older than 2.5 billion years: the cratons.
    Archean,
    Proterozoic,
    /// Younger than 541 million years, including crust reworked at active
    /// boundaries.
    Phanerozoic,
}

impl CrustProvince {
    pub const ALL: [CrustProvince; 4] = [
        CrustProvince::Oceanic,
        CrustProvince::Archean,
        CrustProvince::Proterozoic,
        CrustProvince::Phanerozoic,
    ];

    pub fn name(self) -> &'static str {
        match self {
            CrustProvince::Oceanic => "Oceanic",
            CrustProvince::Archean => "Archean craton",
            CrustProvince::Proterozoic => "Proterozoic",
            CrustProvince::Phanerozoic => "Phanerozoic",
        }
    }

    /// (crust km, felsic fraction, mantle lithosphere km, elevation m) for
    /// a continental province; depletion is solved to float the column at
    /// that elevation. Thick roots need more depletion to stay afloat:
    /// Jordan's (1975) isopycnic cratons. None for oceanic crust.
    fn continental_profile(self) -> Option<(f64, f64, f64, f64)> {
        match self {
            CrustProvince::Oceanic => None,
            CrustProvince::Archean => Some((40.0, 0.55, 180.0, 500.0)),
            CrustProvince::Proterozoic => Some((40.0, 0.5, 130.0, 450.0)),
            CrustProvince::Phanerozoic => Some((33.0, 0.5, 80.0, 350.0)),
        }
    }

    /// The starting column for continental crust of this province.
    pub fn column(self) -> Option<Column> {
        let (crust_km, felsic, mantle_km, elevation_m) = self.continental_profile()?;
        let mut col = Column {
            sediment_km: 0.0,
            crust_km,
            felsic,
            mantle_km,
            depletion: 0.0,
        };
        // Buoyancy is linear in depletion (elevation isn't, across sea
        // level), so solve for the buoyancy that floats at the target.
        let target = SEA_LEVEL_BUOYANCY + elevation_m / 1000.0 * RHO_ASTHENOSPHERE;
        let at_zero = col.buoyancy();
        col.depletion = 1.0;
        let at_one = col.buoyancy();
        col.depletion = ((target - at_zero) / (at_one - at_zero)).clamp(0.0, 1.0);
        Some(col)
    }
}

/// Seafloor elevation (m, negative) for oceanic crust `age` million years
/// old. Young crust follows half-space cooling, depth growing with the
/// square root of age; older crust levels off (Parsons & Sclater 1977).
/// The two meet within about 10 m at 20 Myr.
pub fn ocean_depth_m(age: f64) -> f64 {
    let age = age.max(0.0);
    if age < 20.0 {
        -(2500.0 + 350.0 * age.sqrt())
    } else {
        -(6400.0 - 3200.0 * (-age / 62.8).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3o::Resolution;

    #[test]
    fn depth_follows_age() {
        assert_eq!(ocean_depth_m(0.0), -2500.0);
        // Half-space and plate model agree where they hand over.
        let (young, old) = (-(2500.0 + 350.0 * 20f64.sqrt()), ocean_depth_m(20.0));
        assert!((young - old).abs() < 15.0, "{young} vs {old}");
        // Monotonic deepening, levelling off toward 6400 m.
        let mut last = 0.0;
        for age in (0..=200).step_by(5) {
            let d = ocean_depth_m(age as f64);
            assert!(d <= last);
            last = d;
        }
        assert!(ocean_depth_m(200.0) > -6400.0 && ocean_depth_m(200.0) < -6000.0);
    }

    #[test]
    fn isostasy_matches_earth() {
        // Seafloor floats along the depth-age curve it was calibrated to.
        for age in [0.0, 5.0, 20.0, 60.0, 150.0] {
            let h = Column::oceanic(age, 0.0).elevation_m();
            assert!((h - ocean_depth_m(age)).abs() < 1e-6, "{age} Myr: {h}");
        }
        // A typical continent a little above sea level; Tibet about 5 km up.
        let typical = CrustProvince::Proterozoic.column().unwrap().elevation_m();
        assert!((200.0..600.0).contains(&typical), "{typical}");
        let tibet = Column {
            crust_km: 73.0,
            ..CrustProvince::Proterozoic.column().unwrap()
        };
        assert!((4500.0..6500.0).contains(&tibet.elevation_m()));
        // Sediment raises the seafloor, though by less than its thickness.
        let bare = Column::oceanic(80.0, 0.0).elevation_m();
        let covered = Column::oceanic(80.0, 1.0).elevation_m();
        assert!(
            (300.0..1000.0).contains(&(covered - bare)),
            "{}",
            covered - bare
        );
    }

    #[test]
    fn oceanic_lithosphere_thickens_and_sinks_with_age() {
        // Thickness grows like √age, then levels off near seismic
        // estimates (85 to 95 km under the old Pacific).
        assert_eq!(oceanic_mantle_km(0.0), 0.0);
        let at = |age| oceanic_mantle_km(age);
        assert!((30.0..50.0).contains(&at(20.0)), "{}", at(20.0));
        assert!((75.0..90.0).contains(&at(100.0)), "{}", at(100.0));
        assert!((85.0..100.0).contains(&at(200.0)), "{}", at(200.0));
        // New seafloor floats; it becomes heavier than the mantle under it
        // within 10 to 50 Myr (Cloos 1993); here about 12.
        assert!(Column::oceanic(0.0, 0.0).density_excess() < 0.0);
        let neutral = (0..60)
            .find(|&age| Column::oceanic(age as f64, 0.0).density_excess() > 0.0)
            .unwrap();
        assert!((10..=20).contains(&neutral), "{neutral} Myr");
    }

    #[test]
    fn provinces_float_and_roots_deepen_with_age() {
        let col = |p: CrustProvince| p.column().unwrap();
        let (a, p, ph) = (
            col(CrustProvince::Archean),
            col(CrustProvince::Proterozoic),
            col(CrustProvince::Phanerozoic),
        );
        assert!(a.thickness_km() > 200.0, "{}", a.thickness_km());
        assert!((140.0..=200.0).contains(&p.thickness_km()));
        assert!((100.0..=150.0).contains(&ph.thickness_km()));
        // Isopycnic: the oldest, thickest roots are the most depleted, and
        // all float at modest heights.
        assert!(a.depletion > p.depletion && p.depletion >= ph.depletion);
        for c in [a, p, ph] {
            assert!((300.0..=550.0).contains(&c.elevation_m()), "{c:?}");
            assert!((0.0..=1.0).contains(&c.depletion));
        }
        assert!(CrustProvince::Oceanic.column().is_none());
    }

    #[test]
    fn province_shares() {
        let grid = Grid::new(Resolution::Three);
        let (kind, province) = continents(&grid, &mut Pcg32::new(5));
        let land = kind.iter().filter(|k| k.is_continental()).count() as f64;
        let share = |p| province.iter().filter(|&&x| x == p).count() as f64 / land;
        assert!((share(CrustProvince::Archean) - ARCHEAN_SHARE).abs() < 0.01);
        assert!((share(CrustProvince::Proterozoic) - PROTEROZOIC_SHARE).abs() < 0.01);
        for (k, p) in kind.iter().zip(&province) {
            assert_eq!(k.is_continental(), *p != CrustProvince::Oceanic);
        }
    }

    #[test]
    fn ids_follow_all_order() {
        // The shader looks up province and rock colors by `as u32`.
        for (i, p) in CrustProvince::ALL.iter().enumerate() {
            assert_eq!(*p as usize, i);
        }
        for (i, r) in SurfaceRock::ALL.iter().enumerate() {
            assert_eq!(*r as usize, i);
        }
    }

    #[test]
    fn depletion_for_inverts_buoyancy() {
        let col = CrustProvince::Archean.column().unwrap();
        let b = (RHO_ASTHENOSPHERE - mantle_lithosphere_density(col.depletion)) * col.mantle_km;
        assert!((depletion_for(col.mantle_km, b) - col.depletion).abs() < 1e-12);
        let b = (RHO_ASTHENOSPHERE - crust_density(col.felsic)) * col.crust_km;
        assert!((felsic_for(col.crust_km, b) - col.felsic).abs() < 1e-12);
    }

    #[test]
    fn composition_sets_density() {
        assert_eq!(crust_density(0.0), 2900.0);
        assert_eq!(crust_density(1.0), 2700.0);
        // Depletion offsets the cold root's extra weight.
        assert!(mantle_lithosphere_density(1.0) < mantle_lithosphere_density(0.0));
    }

    #[test]
    fn continents_cover_target_share() {
        let grid = Grid::new(Resolution::Three);
        let (kind, _) = continents(&grid, &mut Pcg32::new(5));
        let land = kind.iter().filter(|k| k.is_continental()).count() as f64;
        let share = land / grid.len() as f64;
        assert!((share - CONTINENTAL_FRACTION).abs() < 0.01, "{share}");
    }
}
