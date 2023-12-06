use anyhow::Result;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::*;
use std::f32::consts::PI;

const TAU: f32 = 2.0 * PI;

fn scale_fn(q: f32, delta: f32) -> Result<f32> {
    Ok(delta / TAU * (2.0 * q - 1.0).asin())
}

#[pyclass]
struct TDigest {
    means: Vec<f32>,
    weights: Vec<i64>,
}

#[pymethods]
impl TDigest {
    #[getter]
    fn total_weight(&self) -> PyResult<i64> {
        Ok(self.weights.iter().sum())
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.means.len())
    }

    fn merge(&mut self, other: &Self) -> PyResult<()> {
        Ok(())
    }

    fn quantile(&self, x: f32) -> PyResult<f32> {
        if self.means.len() < 3 {
            return Ok(0.0);
        }
        let q = x * self.total_weight()? as f32;
        let m = self.means.len();
        let mut cum_weight = 0.0;

        for (i, (m, w)) in self.means.iter().zip(self.weights.iter()).enumerate() {
            let w = *w as f32;
            if cum_weight + w > q {
                let delta;
                if i == 0 {
                    delta = self.means[i + 1] - m;
                } else if i == (m - 1.0) as usize {
                    delta = m - self.means[i - 1];
                } else {
                    delta = (self.means[i + 1] - self.means[i - 1]) / 2.0;
                }
                return Ok(m + ((q - cum_weight) / w - 0.5) * delta);
            }
            cum_weight += w
        }
        Ok(m as f32)
    }

    #[classmethod]
    fn from_arr<'py>(
        _cls: &PyType,
        py: Python<'py>,
        arr: &'py PyArray1<f32>,
        delta: f32,
    ) -> PyResult<Self> {
        let arr = arr.to_owned_array();

        py.allow_threads(|| {
            let n = arr.len();
            let percentile_increment: f32 = 1.0 / n as f32;
            let mut percentile: f32;
            let mut k_lower = scale_fn(0.0, delta)?;
            let mut k_upper: f32;
            let mut start: usize = 0;
            let mut sum: f32 = 0.0;

            // NOTE: the most centroids we could have is equal to `n`
            let mut means: Vec<f32> = Vec::with_capacity(n);
            let mut weights: Vec<i64> = Vec::with_capacity(n);

            for (i, val) in arr.iter().enumerate() {
                percentile = (i as f32 + 1.0) * percentile_increment;
                k_upper = scale_fn(percentile, delta)?;
                sum += val;

                if k_upper - k_lower > 1.0 {
                    let count = i - start + 1;
                    let centroid = sum / (count as f32);
                    means.push(centroid);
                    weights.push(count as i64);
                    k_lower = k_upper;
                    start = i + 1;
                    sum = 0.0;
                }
            }
            Ok(TDigest { means, weights })
        })
    }
}

#[pymodule]
fn tdigest_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<TDigest>()?;
    Ok(())
}
