use std::ffi::CString;

use imgal::prelude::*;
use ndarray::ArrayView3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;

use crate::phasor::batch_calibrated_gs;

/// The AST-net Python parent module.
#[pymodule(name = "ast_net")]
fn ast_net_parent_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let phasor_module = PyModule::new(m.py(), "phasor")?;
    py_import_module("phasor");
    phasor_module.add_function(wrap_pyfunction!(
        phasor_batch_calibrated_gs,
        &phasor_module
    )?)?;
    m.add_submodule(&phasor_module)?;
    Ok(())
}

#[pyfunction]
#[pyo3(name = "batch_calibrated_gs")]
fn phasor_batch_calibrated_gs<'py>(
    py: Python<'py>,
    batch_data: Vec<Bound<'py, PyAny>>,
    calibration_data: Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyArray3<f64>>>> {
    if let Ok(arr_cal) = calibration_data.extract::<PyReadonlyArray3<u8>>() {
        let batch_data: Vec<_> = batch_data
            .iter()
            .map(|v| v.extract::<PyReadonlyArray3<u8>>().map_err(PyErr::from))
            .collect::<PyResult<Vec<_>>>()?;
        let views: Vec<ArrayView3<u8>> = batch_data.iter().map(|v| v.as_array()).collect();
        batch_calibrated_gs(&views, arr_cal.as_array())
            .map(|output| output.into_iter().map(|v| v.into_pyarray(py)).collect())
            .map_err(imgal_error_to_pyerr)
    } else if let Ok(arr_cal) = calibration_data.extract::<PyReadonlyArray3<u16>>() {
        let batch_data: Vec<_> = batch_data
            .iter()
            .map(|v| v.extract::<PyReadonlyArray3<u16>>().map_err(PyErr::from))
            .collect::<PyResult<Vec<_>>>()?;
        let views: Vec<ArrayView3<u16>> = batch_data.iter().map(|v| v.as_array()).collect();
        batch_calibrated_gs(&views, arr_cal.as_array())
            .map(|output| output.into_iter().map(|v| v.into_pyarray(py)).collect())
            .map_err(imgal_error_to_pyerr)
    } else if let Ok(arr_cal) = calibration_data.extract::<PyReadonlyArray3<u64>>() {
        let batch_data: Vec<_> = batch_data
            .iter()
            .map(|v| v.extract::<PyReadonlyArray3<u64>>().map_err(PyErr::from))
            .collect::<PyResult<Vec<_>>>()?;
        let views: Vec<ArrayView3<u64>> = batch_data.iter().map(|v| v.as_array()).collect();
        batch_calibrated_gs(&views, arr_cal.as_array())
            .map(|output| output.into_iter().map(|v| v.into_pyarray(py)).collect())
            .map_err(imgal_error_to_pyerr)
    } else if let Ok(arr_cal) = calibration_data.extract::<PyReadonlyArray3<i64>>() {
        let batch_data: Vec<_> = batch_data
            .iter()
            .map(|v| v.extract::<PyReadonlyArray3<i64>>().map_err(PyErr::from))
            .collect::<PyResult<Vec<_>>>()?;
        let views: Vec<ArrayView3<i64>> = batch_data.iter().map(|v| v.as_array()).collect();
        batch_calibrated_gs(&views, arr_cal.as_array())
            .map(|output| output.into_iter().map(|v| v.into_pyarray(py)).collect())
            .map_err(imgal_error_to_pyerr)
    } else if let Ok(arr_cal) = calibration_data.extract::<PyReadonlyArray3<f32>>() {
        let batch_data: Vec<_> = batch_data
            .iter()
            .map(|v| v.extract::<PyReadonlyArray3<f32>>().map_err(PyErr::from))
            .collect::<PyResult<Vec<_>>>()?;
        let views: Vec<ArrayView3<f32>> = batch_data.iter().map(|v| v.as_array()).collect();
        batch_calibrated_gs(&views, arr_cal.as_array())
            .map(|output| output.into_iter().map(|v| v.into_pyarray(py)).collect())
            .map_err(imgal_error_to_pyerr)
    } else if let Ok(arr_cal) = calibration_data.extract::<PyReadonlyArray3<f64>>() {
        let batch_data: Vec<_> = batch_data
            .iter()
            .map(|v| v.extract::<PyReadonlyArray3<f64>>().map_err(PyErr::from))
            .collect::<PyResult<Vec<_>>>()?;
        let views: Vec<ArrayView3<f64>> = batch_data.iter().map(|v| v.as_array()).collect();
        batch_calibrated_gs(&views, arr_cal.as_array())
            .map(|output| output.into_iter().map(|v| v.into_pyarray(py)).collect())
            .map_err(imgal_error_to_pyerr)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, i64, f32, and f64.",
        ))
    }
}

/// Convert a ImgalError into a RuntimeError PyErr
///
/// This is a quick/easy way to map imgal's errors.
pub fn imgal_error_to_pyerr(err: ImgalError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

/// Add a child module to Python's sys.modules dict.
///
/// # Description
///
/// This function manually adds a given module to Python's sys.modules
/// dict. This enables imports like `import ast_net.phasor as phr`.
///
/// # Arguments
///
/// * `module_name` - The name of the module to add to sys.modules.
fn py_import_module(module_name: &str) {
    let import_cmd = format!(
        "import sys; sys.modules['ast_net.{}'] = '{}'",
        module_name, module_name
    );
    let c_str_cmd =
        CString::new(import_cmd).expect("Failed to create 'CString' module import command.");
    Python::attach(|py| {
        py.run(c_str_cmd.as_c_str(), None, None).unwrap();
    });
}
