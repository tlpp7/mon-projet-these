import numpy as np
import matplotlib.pyplot as plt

def fourier_neural_operator_layer(input_signal, weights_fourier, alpha_regularizer):
    """
    Simule le passage d'un champ de solution dans une couche FNO 
    avec filtrage des hautes fréquences et projection proximale.
    """
    # 1. Passage dans l'espace des fréquences (FFT)
    fourier_modes = np.fft.rfft(input_signal)
    
    # 2. Multiplication par les poids du réseau (Filtrage neuronal)
    filtered_modes = fourier_modes * weights_fourier
    
    # 3. Retour dans l'espace physique (IFFT)
    output_signal = np.fft.irfft(filtered_modes, n=len(input_signal))
    
    # 4. Étape proximale non linéaire (Filtre de Nash injecté par la thèse)
    return np.maximum(0, output_signal - alpha_regularizer)

if __name__ == "__main__":
    n_space = 256
    x = np.linspace(-np.pi, np.pi, n_space)
    
    # Signal d'entrée complexe (Profil de matériau poreux rugueux)
    np.random.seed(42)
    signal_brut = np.sin(2 * x) + 0.5 * np.cos(5 * x) + np.random.normal(0, 0.2, n_space)
    
    # Poids de Fourier neuronaux (Filtre passe-bas idéal appris par le réseau)
    weights = np.zeros(n_space // 2 + 1)
    weights[:12] = 1.0  # On ne conserve que les 12 premiers modes structurels
    
    signal_reconstruit = fourier_neural_operator_layer(signal_brut, weights, alpha_regularizer=0.1)
    
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, signal_brut, 'r:', alpha=0.5, label="Signal Inverse Bruité (Entrée)")
    plt.plot(x, signal_reconstruit, 'b-', linewidth=2, label="Sortie C-FNO (Topologiquement Propre)")
    plt.title("22. Filtrage et Régularisation via Opérateur Neuronal (C-FNO)")
    plt.xlabel("Domaine Continu Spatial")
    plt.ylabel("Amplitude de la Solution")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()
