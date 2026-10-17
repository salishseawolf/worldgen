use crate::geo::{LonLat, Polygon, same_point};

/// Polygon drawing state: finished polygons plus the ring being drawn.
#[derive(Default)]
pub struct Drawing {
    pub polygons: Vec<Polygon>,
    pub current: Vec<LonLat>,
}

impl Drawing {
    /// Add a vertex to the ring in progress. Repeats of the last vertex are
    /// ignored, which also absorbs the second click of a double-click.
    pub fn add_vertex(&mut self, p: LonLat) {
        if self.current.last().is_some_and(|&last| same_point(last, p)) {
            return;
        }
        self.current.push(p);
    }

    pub fn undo_vertex(&mut self) {
        self.current.pop();
    }

    pub fn cancel(&mut self) {
        self.current.clear();
    }

    /// Finish the ring in progress. Returns false (and keeps drawing) if it
    /// doesn't have 3 distinct vertices yet.
    pub fn close(&mut self) -> bool {
        match Polygon::from_ring(self.current.clone()) {
            Some(polygon) => {
                self.polygons.push(polygon);
                self.current.clear();
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_close_and_dedupe() {
        let mut d = Drawing::default();
        d.add_vertex(LonLat::new(0.0, 0.0));
        d.add_vertex(LonLat::new(10.0, 0.0));
        assert!(!d.close(), "two vertices can't close");
        d.add_vertex(LonLat::new(10.0, 10.0));
        d.add_vertex(LonLat::new(10.0, 10.0));
        assert_eq!(d.current.len(), 3);
        assert!(d.close());
        assert_eq!(d.polygons.len(), 1);
        assert!(d.current.is_empty());
    }

    #[test]
    fn undo_and_cancel() {
        let mut d = Drawing::default();
        d.add_vertex(LonLat::new(0.0, 0.0));
        d.add_vertex(LonLat::new(5.0, 5.0));
        d.undo_vertex();
        assert_eq!(d.current, vec![LonLat::new(0.0, 0.0)]);
        d.cancel();
        assert!(d.current.is_empty());
    }
}
