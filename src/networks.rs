use burn::backend::Wgpu;
use burn::prelude::*;
use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2}; use pyo3::prelude::*;

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

    parent_module.add_submodule(&stardist_module)
}

/// run StarDist2D on a (512, 512) image.
#[pyfunction]
#[pyo3(name = "stardist_2d")]
pub fn py_stardist_2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray2<f32>> {
    run_stardist_2d(data.as_array()).into_pyarray(py)
}

/// actually run it lol
#[inline]
fn run_stardist_2d(data: ArrayView2<f32>) -> Array2<f32> {
    // setup the model
    let device = Default::default();
    let stardist_model = Model::<Backend>::default();

    // create tenor from input array
    let tensor = Tensor::<Backend, 1>::from_floats(
        data.into_owned().into_flat().as_slice().unwrap(),
        &device,
    );
    let (a, _b) = stardist_model.forward(tensor);
    let d: Vec<f32> = a.into_data().into_vec().unwrap();
    Array2::from_shape_vec((512, 512), d).expect("Data reshape failed.")
}
