use numpy::ndarray::{ArrayViewD};
use numpy::PyArray;
use numpy::{
    PyArrayDyn, PyReadonlyArrayDyn,
};
use pyo3::{
    pymodule,
    types::{PyModule},
    PyResult, Python,
};
use std::{f32::consts::PI};
use num_traits::{Float};

const TAU: f32 = 2.0 * PI;

fn scale_fn(q: f32, delta: f32) -> f32 {
    delta / TAU * (2.0 * q - 1.0).asin()
}

#[derive(Debug)]
struct TDigest {
    means: Vec<f32>,
    weights: Vec<i64>
}


impl TDigest {
    fn from_vec(mut vec: Vec<f32>, delta: f32) -> TDigest
    {
        // vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // glidesort::sort_by(&mut vec|a, b| a.partial_cmp(b).unwrap());

        let n = vec.len();
        let percentile_increment: f32 = 1.0 / n as f32;
        let mut percentile: f32;
        let mut k_lower = scale_fn(0.0, delta);
        let mut k_upper: f32;
        let mut start: usize = 0;
        let mut sum: f32 = 0.0;

        let mut means: Vec<f32> = Vec::new();
        let mut weights: Vec<i64> = Vec::new();

        for (i, val) in vec.iter().enumerate() {
            percentile = (i as f32 + 1.0) * percentile_increment;
            k_upper = scale_fn(percentile, delta);
            sum += val;

            if k_upper - k_lower > 1.0 {
                let count = i - start + 1;
                let centroid = (sum / (count as f32)) as f32;
                means.push(centroid);
                weights.push(count as i64);
                k_lower = k_upper;
                start = i + 1;
                sum = 0.0;
            }
        }
        TDigest{means: means, weights: weights}
    }
}



#[pymodule]
fn tdigest_ext<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    fn create_from_array(delta: f32, x: ArrayViewD<'_, f32>) -> (Vec<f32>, Vec<i64>) {
        let vec = x.into_owned().into_raw_vec();
        let tdigest = TDigest::from_vec(vec.clone(), delta);
        (tdigest.means, tdigest.weights)
    }

    #[pyfn(m)]
    #[pyo3(name = "create_from_array")]
    fn create_from_array_py<'py>(
        py: Python<'py>,
        delta: f32,
        x: PyReadonlyArrayDyn<'py, f32>,
    ) -> (&'py PyArrayDyn<f32>, &'py PyArrayDyn<i64>) {
        let x = x.as_array();
        let (means, weights) = py.allow_threads(|| create_from_array(delta, x));
        let means = PyArray::from_vec(py, means);
        let weights = PyArray::from_vec(py, weights);
        (means.to_dyn(), weights.to_dyn())
    }

    Ok(())
}
