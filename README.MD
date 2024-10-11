I. Introduction

The paper begins by discussing the importance of modifying gravitational theories to interpret cosmological data. It introduces the concept of f(Q) gravity, which is based on non-metricity in a space without curvature and torsion.

Key points:
1. General Relativity (GR) uses the Ricci scalar R with the Levi-Civita connection.
2. f(R) theory modifies GR by using a function of R.
3. Other connections lead to different theories:
   - Weitzenbock connection: Teleparallel Equivalent of General Relativity (TEGR)
   - Torsion-free, flat connection: Symmetric Teleparallel Equivalent of GR (STEGR)
4. f(Q) theory is a modification of STEGR.

II. Mathematical Formulation

This section introduces the fundamental mathematical concepts of f(Q) gravity.

1. Non-metricity tensor:
   $$Q_{\lambda\mu\nu} = \nabla_\lambda g_{\mu\nu}$$

2. Traces of the non-metricity tensor:
   $$Q_\lambda = Q_{\lambda\mu\nu} g^{\mu\nu}, \quad \tilde{Q}_\nu = Q_{\lambda\mu\nu} g^{\lambda\mu}$$

3. Superpotential tensor:
   $$P^\lambda_{\mu\nu} = \frac{1}{4}\left(-2L^\lambda_{\mu\nu} + Q^\lambda g_{\mu\nu} - \tilde{Q}^\lambda g_{\mu\nu} - \frac{1}{2} \delta^\lambda_\mu Q_\nu - \frac{1}{2} \delta^\lambda_\nu Q_\mu\right)$$

4. Disformation tensor:
   $$L^\lambda_{\mu\nu} = \frac{1}{2}\left(Q^\lambda_{\mu\nu} - Q_\mu{}^\lambda{}_\nu - Q_\nu{}^\lambda{}_\mu\right)$$

5. Relation between affine connection and Levi-Civita connection:
   $$\Gamma^\lambda_{\mu\nu} = \hat{\Gamma}^\lambda_{\mu\nu} + L^\lambda_{\mu\nu}$$

6. Non-metricity scalar:
   $$Q = Q_{\lambda\mu\nu} P^{\lambda\mu\nu} = \frac{1}{4}\left(-Q_{\lambda\mu\nu} Q^{\lambda\mu\nu} + 2Q_{\lambda\mu\nu} Q^{\mu\lambda\nu} + Q_\lambda Q^\lambda - 2Q_\lambda \tilde{Q}^\lambda\right)$$

7. Action in f(Q) theory:
   $$S = \frac{1}{2\kappa}\int f(Q)\sqrt{-g} d^4x + \int L_M\sqrt{-g} d^4x$$

8. Field equations:
   $$2\sqrt{-g} \nabla_\lambda(\sqrt{-g}FP^\lambda{}_{\mu\nu}) - \frac{1}{2} fg_{\mu\nu} + F\left(P_{\nu\rho\sigma} Q^\rho{}_\sigma{}_\mu - 2P_{\rho\sigma\mu} Q^{\rho\sigma}{}_\nu\right) = \kappa T^m_{\mu\nu}$$

   Where $F = \frac{df}{dQ}$ and $T^m_{\mu\nu}$ is the stress-energy tensor of perfect fluid:
   $$T^m_{\mu\nu} = (p + \rho)u_\mu u_\nu + pg_{\mu\nu}$$

9. Covariant formulation:
   $$F \hat{G}_{\mu\nu} + \frac{1}{2} g_{\mu\nu}(FQ - f) + 2F' P^\lambda{}_{\mu\nu} \hat{\nabla}_\lambda Q = \kappa T^m_{\mu\nu}$$

   Which can be expressed in GR equivalent form:
   $$\hat{G}_{\mu\nu} = \frac{\kappa}{F} T_{\mu\nu} = \frac{\kappa}{F} T^m_{\mu\nu} + \kappa T^{DE}_{\mu\nu}$$

   Where:
   $$\kappa T^{DE}_{\mu\nu} = \frac{1}{F}\left(\frac{1}{2} g_{\mu\nu}(f - QF) - 2F' \hat{\nabla}_\lambda QP^\lambda{}_{\mu\nu}\right)$$

III. Equations of Motion

This section derives the equations of motion for f(Q) gravity in a Friedmann-Lemaitre-Robertson-Walker (FLRW) metric.

1. FLRW metric:
   $$dS^2 = -dt \otimes dt + a(t)^2 \left(dr \otimes dr + r^2 d\theta \otimes d\theta + r^2 \sin^2 \theta d\phi \otimes d\phi\right)$$

2. Non-vanishing components of the affine connection:
   Various components are listed, including $\Gamma^t_{tt} = \gamma + \frac{\dot{\gamma}}{\gamma}$, $\Gamma^r_{tr} = \gamma$, etc.

3. Non-metricity scalar:
   $$Q = -6H^2 + 9\gamma H + 3\dot{\gamma}$$

4. Friedmann-like equations:
   $$\kappa\rho = \frac{1}{2}f + \left(3H^2 - \frac{Q}{2}\right)F + \frac{3}{2}\dot{Q}\gamma F'$$
   $$\kappa p = -\frac{1}{2}f + \left(-2\dot{H} - 3H^2 + \frac{Q}{2}\right)F + \frac{\dot{Q}}{2}(-4H + 3\gamma)F'$$

5. Proposed new parameterization of the Hubble function:
   $$H(z) = H_0(z + 1)^n + \beta\left[1 - (z + 1)^n\right]$$

IV. Data and Methodology

This section describes the statistical analysis used to constrain the model parameters.

1. Bayesian analysis and MCMC techniques are used.
2. Data from Cosmic Chronometers, Pantheon+SH0ES, and Baryonic Acoustic Oscillations are used.
3. The total $\chi^2$ function is minimized:
   $$\chi^2_{total} = \chi^2_{CC} + \chi^2_{SN} + \chi^2_{BAO}$$

V. Cosmological f(Q) Models

This section examines specific f(Q) models:

1. Model-I: $f(Q) = Q + \eta e^{\alpha Q}$
2. Model-II: $f(Q) = Q + \eta \log(\alpha Q)$
3. Model-III: $f(Q) = Q + \eta Q^\alpha$
4. Model-IV: $f(Q) = Q + \eta Q^{-1}$
5. Model-V: $f(Q) = Q + \eta Q^2$

These models are analyzed using energy conditions and thermodynamical stability.

VI. Conclusion

The paper concludes by summarizing the key findings:

1. The proposed Hubble function parameterization is consistent with observational data.
2. Various f(Q) models are found to be viable under energy conditions and thermodynamical stability analysis.
3. The non-coincident formalism of f(Q) gravity provides distinct results from f(T) gravity.
4. Some limitations of the f(Q) framework are noted, but the authors argue it still has interesting theoretical features.

This paper provides a comprehensive analysis of f(Q) gravity in a non-coincident formalism, testing its viability against observational data and theoretical constraints. It contributes to our understanding of how modified gravity theories might explain cosmic acceleration without the need for dark energy.