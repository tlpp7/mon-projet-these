// ==============================================================================
// Kernel de Transport Optimal Entropique (Algorithme de Sinkhorn en Rust)
// ==============================================================================

use pyo3.prelude::*;
use ndarray::prelude::*;
use ndarray::Zip;

#[pyfunction]
fn fast_sinkhorn_w2(
    a: Vec<f64>, 
    b: Vec<f64>, 
    cost_matrix: Vec<Vec<f64>>, 
    reg: f64, 
    max_iter: usize
) -> PyResult<f64> {
    let n = a.len();
    let m = b.len();

    // Matrice de Gibbs K = exp(-C / reg)
    let mut k_matrix = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            k_matrix[i][j] = (-cost_matrix[i][j] / reg).exp();
        }
    }

    // Vecteurs de mise à l'échelle (scaling factors)
    let mut u = vec![1.0 / (n as f64); n];
    let mut v = vec![1.0 / (m as f64); m];

    // Itérations de Sinkhorn-Knopp
    for _ in 0..max_iter {
        // u = a / (K * v)
        for i in 0..n {
            let mut kv = 0.0;
            for j in 0..m {
                kv += k_matrix[i][j] * v[j];
            }
            u[i] = a[i] / (kv + 1e-12);
        }

        // v = b / (K^T * u)
        for j in 0..m {
            let mut kt_u = 0.0;
            for i in 0..n {
                kt_u += k_matrix[i][j] * u[i];
            }
            v[j] = b[j] / (kt_u + 1e-12);
        }
    }

    // Calcul de la distance W2 régularisée
    let mut total_cost = 0.0;
    for i in 0..n {
        for j in 0..m {
            let gamma_ij = u[i] * k_matrix[i][j] * v[j];
            total_cost += gamma_ij * cost_matrix[i][j];
        }
    }

    Ok(total_cost)
}

#[pymodule]
fn rust_optimal_transport(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_sinkhorn_w2, m)?)?;
    Ok(())
}
