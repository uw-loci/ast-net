use std::ffi::CString;

use pyo3::prelude::*;

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
fn phasor_batch_calibrated_gs() {
    todo!();
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
