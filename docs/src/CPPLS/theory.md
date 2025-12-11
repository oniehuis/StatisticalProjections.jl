## Canonical Powered Partial Least Squares (CPPLS)

Canonical Powered Partial Least Squares (CPPLS) is a supervised projection method for regression and classification. Its purpose is to extract latent components that summarize the predictor matrix \(X \in \mathbb{R}^{n \times p}\) in a way that best reflects the structure in a multivariate response matrix \(Y \in \mathbb{R}^{n \times q}\). The method extends standard PLS in three important ways. First, it allows multiple response variables, including both primary responses that one wishes to predict and optional auxiliary responses that guide the extraction of components. Second, it incorporates a power parameter \(\gamma\) that controls the balance between predictor variance and predictor–response correlation, giving the user explicit control over how strongly the model should emphasize correlation structure. Third, CPPLS can operate with a vector of non-negative sample weights, allowing some observations to contribute more or less to the fitted model. This is useful when classes are unbalanced, when some samples are more reliable, or when experimental design considerations suggest that certain samples should carry greater influence.

Each CPPLS component is extracted in two conceptual stages. First, the predictors are projected onto supervised directions, one for each column of \(Y\), using the \(\gamma\)-controlled mixture of weighted predictor variance and weighted predictor–response correlation. Second, a canonical correlation analysis (CCA) determines how these supervised directions should be linearly combined into a single latent variable that is optimally aligned with the primary responses. Auxiliary responses and observation weights enter the computation of supervised directions in the first stage, shaping the latent space that is subsequently analyzed by CCA, while the CCA itself is guided solely by the primary responses under the same weighting structure.

To begin, \(X\) and \(Y\) are centered (and optionally scaled) using the supplied sample weights. If \(w_i\) is the weight of sample \(i\) and the weights are normalized to sum to one, the weighted mean of a variable \(x\) becomes
```math
\bar{x} = \sum_i w_i x_i ,
```
and weighted inner products
```math
\langle u, v \rangle_w = \sum_i w_i u_i v_i
```
define all covariances. CPPLS computes, for each predictor in \(X\), its weighted variance and its weighted covariance with each column of \(Y\). These two quantities are blended according to the power parameter \(\gamma\). When \(\gamma\) is small, predictor variance carries more influence; when \(\gamma\) approaches one, the emphasis shifts to predictor–response correlation. This results in a supervised weight matrix
```math
W_0(\gamma) \in \mathbb{R}^{p \times q},
```
where \(p\) is the number of predictors and \(q\) is the number of response columns, including auxiliary ones. Each column of \(W_0(\gamma)\) is a supervised direction in the original predictor space. If auxiliary responses are supplied, they contribute additional columns and thereby enrich the set of supervised directions. Because the construction of \(W_0(\gamma)\) uses the sample weights, heavily weighted samples exert proportionally greater influence on the supervised compression.

Multiplying the predictor matrix by this weight matrix gives the \(\gamma\)-dependent supervised compression
```math
Z(\gamma) = X W_0(\gamma),
```
which has the same number of samples as \(X\) but one column for each response variable. Every entry in \(Z(\gamma)\) is a single summary value for a sample in the supervised direction associated with a response column. Auxiliary responses provide additional supervised views of the data, and sample weights ensure that classes or samples with higher weights have greater influence on how these supervised directions are shaped.

To determine the optimal balance between variance and correlation, CPPLS evaluates a grid of \(\gamma\)-values. For each candidate \(\gamma\), CPPLS computes \(Z(\gamma)\) and performs a weighted canonical correlation analysis between \(Z(\gamma)\) and the primary responses. Only the first canonical correlation is kept and serves as a score indicating how well that \(\gamma\) captures structure relevant to the primary response. The optimal value \(\gamma_{\mathrm{best}}\) is the one maximizing this weighted canonical correlation. This step does not yet produce components or deflate the data; it merely evaluates each candidate \(\gamma\) under identical conditions.

Once the optimal \(\gamma\) has been determined, CPPLS recomputes
```math
Z = Z(\gamma_{\mathrm{best}})
```
and performs a full weighted CCA between this matrix and the primary response columns of \(Y\). The result is a canonical direction \(a\) in the \(Z\)-space and a corresponding direction \(b\) in the primary response space. The direction \(a\) specifies how to combine the supervised directions in \(Z\) into one axis that maximally correlates with the primary responses. Auxiliary responses influence this direction indirectly, since they have already shaped the supervised directions in \(W_0(\gamma_{\mathrm{best}})\). The canonical direction is then mapped back into the predictor space through
```math
w = W_0(\gamma_{\mathrm{best}})\, a ,
```
producing the final CPPLS weight vector. This vector lies in the original predictor space and defines the direction used to compute the component score
```math
t = X w .
```

The component score \(t\) acts as a latent one-dimensional slider: each sample receives a coordinate \(t_i\), and moving along this latent axis corresponds to sliding along the component in predictor space. The relationship between the component and the original variables is captured by the loadings. The weighted X-loading is given by
```math
p = \frac{X^\top W t}{t^\top W t},
```
which is the weighted regression of the predictors on the component. It describes how each predictor changes as one moves along \(t\). The weighted Y-loading
```math
c = \frac{Y^\top W t}{t^\top W t}
```
describes how each response variable—including auxiliary responses when present—varies with the component under the weighting structure.

Deflation removes the part of \(X\) and \(Y\) that can be explained by this component:
```math
X \leftarrow X - t p^\top,\qquad
Y \leftarrow Y - t c^\top .
```

After this deflation, the dominant species structure has been removed from both \(X\) and \(Y\). Because subsequent components are extracted from these residuals, they often describe remaining structured variation—such as seasonal drift or batch effects—rather than additional species separation. This does not impair discrimination: the first component captures the primary class separation, and later components model the remaining confounding structure, improving stability and interpretability.

Sample weighting becomes particularly important in discriminant analysis (CPPLS-DA) when classes are unbalanced or when certain samples are more reliable. For example, if one species has 20 samples and another 60, the unweighted analysis gives the larger class three times the influence on variances and correlations, and thus on the supervised compression. This often causes the dominant component to reflect within-class variation of the majority class rather than the between-class difference one intends to model. By assigning larger weights to minority-class samples and smaller weights to majority-class samples, the total effective weight of each class becomes balanced. All weighted variances, covariances, and correlations then reflect this adjusted influence, and the extracted components focus on genuine between-class separation rather than being overwhelmed by the larger class. Similarly, samples known to be noisy or unreliable can be down-weighted, while representative or carefully controlled samples can be up-weighted, improving the robustness of the extracted components.

A concrete example illustrates the benefit of combining auxiliary responses and sample weighting. Suppose two insect species are analyzed by GC–MS to characterize their cuticular hydrocarbons. The primary task is to discriminate species, but chemical profiles change with season, and species may not be collected uniformly throughout the year. In this situation, the largest variation in \(X\) may reflect seasonal drift rather than species. Even if the response \(Y\) encodes only species, the supervised compression may inadvertently emphasize seasonal structure, because GC–MS peaks that vary with season often correlate indirectly with species when sampling times differ. Including sampling date or season as an auxiliary Y-column provides CPPLS with a supervised direction dedicated to seasonal variation, preventing this structure from leaking into the species component. If sampling imbalance is also present—say one species is collected mostly early in the season and the other mostly later—assigning balanced sample weights prevents the majority class or sampling pattern from dominating the supervised compression. Together, auxiliary \(Y\) and sample weighting produce a far more stable and interpretable latent structure: the first component becomes species-focused, while secondary components capture the residual seasonal drift or other confounders. X-loadings highlight species-specific peaks, while seasonal peaks are isolated to components guided by the auxiliary response.

After all components are extracted, regression coefficients for predicting the primary responses are assembled using
```math
B =
W_{\mathrm{comp}}
\left( P^\top W_{\mathrm{comp}} \right)^{-1}
C_{\mathrm{primary}}^\top ,
```
where \(W_{\mathrm{comp}}\) contains the component weight vectors, \(P\) the corresponding X-loadings, and \(C_{\mathrm{primary}}\) the primary Y-loadings. Auxiliary responses influence the latent components but do not appear in the final regression model unless explicitly designated as prediction targets.

In summary, CPPLS combines three complementary forms of supervision: the power parameter \(\gamma\) that controls the balance between variance and correlation, auxiliary responses that provide additional structured guidance for the supervised compression, and sample weights that ensure appropriate influence of different samples or classes. Together, these features allow CPPLS to build stable, interpretable, and discriminative models even in complex, high-dimensional, and confounded data settings.
