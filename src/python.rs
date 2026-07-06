use std::collections::HashMap;
use std::ffi::CString;

use imgal::prelude::*;
use ndarray::ArrayView3;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;

use crate::phasor::batch_segment_gs;

/// The AST-net Python parent module.
#[pymodule(name = "ast_net")]
fn ast_net_parent_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let phasor_module = PyModule::new(m.py(), "phasor")?;
    py_import_module("phasor");
    phasor_module.add_function(wrap_pyfunction!(phasor_batch_segment_gs, &phasor_module)?)?;
    m.add_submodule(&phasor_module)?;
    Ok(())
}

/// TODO
#[pyfunction]
#[pyo3(name = "batch_segment_gs")]
fn phasor_batch_segment_gs<'py>(
    py: Python<'py>,
    batch_data: Vec<Bound<'py, PyAny>>,
    calibration_data: Bound<'py, PyAny>,
    calibration_tau: f64,
    period: f64,
) -> PyResult<Vec<HashMap<u64, Py<PyArray2<f64>>>>> {
    // TODO use PyReadonlyArray3<u16> as input?
    if let Ok(arr_cal) = calibration_data.extract::<PyReadonlyArray3<u16>>() {
        let batch_data: Vec<_> = batch_data
            .iter()
            .map(|v| v.extract::<PyReadonlyArray3<u16>>().map_err(PyErr::from))
            .collect::<PyResult<Vec<_>>>()?;
        let views: Vec<ArrayView3<u16>> = batch_data.iter().map(|v| v.as_array()).collect();
        batch_segment_gs(&views, arr_cal.as_array(), calibration_tau, period)
            .map(|output| {
                output
                    .into_iter()
                    .map(|m| {
                        m.into_iter()
                            .map(|(k, v)| (k, v.into_pyarray(py).unbind()))
                            .collect()
                    })
                    .collect()
            })
            .map_err(imgal_error_to_pyerr)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, only u16 is supported.",
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
