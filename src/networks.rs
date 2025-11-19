use std::ffi::CString;

use burn::backend::Wgpu;
use burn::prelude::*;
use imgal::image::percentile_normalize;
use imgal::traits::numeric::AsNumeric;
use ndarray::{Array2, Array3, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::models::stardist::Model;

type Backend = Wgpu<f32, i32>;

/// Parent module
#[pymodule(name = "ast_net")]
fn ast_net_parent_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_stardist_module(m)?;
    Ok(())    
}

/// Create stardist submodule
pub fn register_stardist_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let stardist_module = PyModule::new(parent_module.py(), "stardist")?;
    stardist_module.add_function(wrap_pyfunction!(
        py_stardist_2d,
        &stardist_module
    )?)?;
    py_import_module("stardist");

    parent_module.add_submodule(&stardist_module)
}

/// Run StarDist2D on an image.
#[pyfunction]
#[pyo3(name = "stardist_2d")]
pub fn py_stardist_2d<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray3<f32>>)> {
    if let Ok(arr) = data.extract::<PyReadonlyArray2<u8>>() {
        let (prob_arr, dist_arr) = run_stardist_2d(arr.as_array());
        return Ok((prob_arr.into_pyarray(py), dist_arr.into_pyarray(py)));
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<u16>>() {
        let (prob_arr, dist_arr) = run_stardist_2d(arr.as_array());
        return Ok((prob_arr.into_pyarray(py), dist_arr.into_pyarray(py)));
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<u64>>() {
        let (prob_arr, dist_arr) = run_stardist_2d(arr.as_array());
        return Ok((prob_arr.into_pyarray(py), dist_arr.into_pyarray(py)));
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<f32>>() {
        let (prob_arr, dist_arr) = run_stardist_2d(arr.as_array());
        return Ok((prob_arr.into_pyarray(py), dist_arr.into_pyarray(py)));
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<f64>>() {
        let (prob_arr, dist_arr) = run_stardist_2d(arr.as_array());
        return Ok((prob_arr.into_pyarray(py), dist_arr.into_pyarray(py)));
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64."
        ))
    }
}

/// StarDist2D
#[inline]
fn run_stardist_2d<T>(data: ArrayView2<T>) -> (Array2<f32>, Array3<f32>)
where
    T: AsNumeric,
{
    // setup the model
    let device = Default::default();
    let stardist_model = Model::<Backend>::default();

    // create tenor from input array
    // normalize the data...maybe expose as adjustable parameter?
    let norm_data = percentile_normalize(data, 1.0, 99.8, None, None);
    let norm_data = norm_data.mapv(|v| v as f32);
    // todo: convert to f32
    let tensor = Tensor::<Backend, 1>::from_floats(
        norm_data.into_flat().as_slice().unwrap(),
        &device,
    );
    let (a, b) = stardist_model.forward(tensor);
    let result_a: Vec<f32> = a.into_data().into_vec().unwrap();
    let result_b: Vec<f32> = b.into_data().into_vec().unwrap();
    let arr_a = Array2::from_shape_vec((256, 256), result_a).expect("Tensor data reshape failed.");
    let arr_b = Array3::from_shape_vec((256, 256, 32), result_b).expect("Tensor data reshape failed.");

    (arr_a, arr_b)
}

/// Add a child module to Python's sys.modules dict.
///
/// # Description
///
/// This function manually adds a given module to Python's sys.modules
/// dict. This enables imports like `import ast_net.stardist as star`.
///
/// # Arguments
///
/// * `module_name` - The name of the module to add to sys.modules.
fn py_import_module(module_name: &str) {
    let import_cmd = format!(
        "import sys; sys.modules['imgal.{}'] = '{}'",
        module_name, module_name
    );
    let c_str_cmd =
        CString::new(import_cmd).expect("Failed to create 'CString' module import command.");
    Python::attach(|py| {
        py.run(c_str_cmd.as_c_str(), None, None).unwrap();
    });
}
