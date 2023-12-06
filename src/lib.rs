use anyhow::Result;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::f32::consts::PI;

const TAU: f32 = 2.0 * PI;

fn scale_fn(q: f32, delta: f32) -> Result<f32> {
    Ok(delta / TAU * (2.0 * q - 1.0).asin())
}

struct TDigest {
    means: Vec<f32>,
    weights: Vec<i64>,
}

impl TDigest {
    fn from_arr(arr: &ArrayView1<f32>, delta: f32) -> Result<TDigest> {
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
    }
}

#[pyfunction]
fn create_from_array<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<f32>,
    delta: f32,
) -> PyResult<(&'py PyArray1<f32>, &'py PyArray1<i64>)> {
    let vec = arr.as_array();
    let tdigest = py.allow_threads(|| TDigest::from_arr(&vec, delta))?;
    Ok((
        PyArray1::from_vec(py, tdigest.means),
        PyArray1::from_vec(py, tdigest.weights),
    ))
}

#[pymodule]
fn tdigest_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_from_array, m)?)?;
    Ok(())
}
