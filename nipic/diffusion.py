
import dipy.io
from dipy.segment.mask import median_otsu
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import fractional_anisotropy, color_fa, lower_triangular

def compute_FA(a, extra)
    # http://nipy.org/dipy/examples_built/quick_start.html#example-quick-start
    logger.info('Computing DTI mask...')
    maskdata, mask = median_otsu(a, 3, 1, True, dilate=2) #TODO: improve this
    logger.info('Computing DTI gradient table...')
    gtab = gradient_table(extra['b_values'], extra['grad_table'])
    logger.info('Fitting DTI tensor model...')
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)
    logger.info('Computing fractional anisotropy...')
    fa = fractional_anisotropy(tenfit.evals)
    fa[np.isnan(fa)] = 0
    return fa

def reduce_dti(a, extra):
    return ('FA', [compute_FA(a, extra)])
