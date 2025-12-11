import os.path as op
import statsmodels.api as sm
from pingouin import mediation_analysis

import graphviz
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

def linear_regression(df, predictor, outcome, covariables=None, interactions=None,
                      center_vars=None, show_std_coeffs=True, ylabel=None, xlabel=None, fig=None,
                      scatter_symbol='o', scatter_label=None, reg_color='r',
                      fig_dir=None, report_dir=None, file_suffix=''):

    if covariables is None:
        covariables = []

    if interactions is None:
        interactions = []

    if center_vars is None:
        center_vars = []

    if ylabel is None:
        ylabel = outcome

    if xlabel is None:
        xlabel = predictor

    df = df.copy()

    for center_var in center_vars:
        df[center_var] = df[center_var] - df[center_var].mean()

    reg_vars = [predictor] + covariables

    interaction_vars = []
    for interaction in interactions:
        interaction_var = '_X_'.join(interaction)
        df[interaction_var] = df[interaction[0]] * df[interaction[1]]
        interaction_vars.append(interaction_var)

    m_na = pd.isna(df[[outcome] + reg_vars + interaction_vars]).any(axis=1)
    if m_na.any():
        print('WARNING: fitler %d rows because they contain nans' % m_na.sum())
        df = df[~m_na]

    df = df.sort_values(predictor)

    pred = sm.add_constant(df[reg_vars + interaction_vars])

    ols_model = sm.OLS(df[outcome], pred)
    estimate = ols_model.fit()

    # print(estimate.summary())

    #predicted = estimate.get_prediction()
    #iv_l = predicted.summary_frame()["obs_ci_lower"]
    #iv_u = predicted.summary_frame()["obs_ci_upper"]

    covar_fit = ''
    if show_std_coeffs:
        #y_std = df[outcome].std()
        estimates = {c:v*(df[c].std() if c !='const' else 1) for c,v in estimate.params.items()}
    else:
        estimates = estimate.params

    if len(covariables) > 0:
        covar_fit = ' ' + ' '.join('%+1.2f*%s' %(estimates[c], c) for c in covariables)

    eq_label = ('%s = %s*%s%s %+1.2f (%s)'
                %(outcome, format_factor(estimates[predictor]),
                  predictor, covar_fit, estimates['const'],
                  format_pvalue(estimate.pvalues[predictor], latex=True)))

    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        ax = fig.gca()

    x = df[predictor]
    y = df[outcome]
    ax.plot(x, y, scatter_symbol, label=scatter_label)

    x_bins = np.array([x.min(), x.max()])
    ci = estimate.conf_int()
    iv_l = (ci.loc[predictor, 0] * x_bins + estimate.params['const'] +
            sum([estimate.params[c]*df[c].median()
                 for c in ci.index if c not in {predictor, 'const'}]))
    iv_u = (ci.loc[predictor, 1] * x_bins + estimate.params['const'] +
            sum([estimate.params[c]*df[c].median()
                 for c in ci.index if c not in {predictor, 'const'}]))
    ax.fill_between(x_bins, iv_l, iv_u, color='grey', alpha=0.1)
    plot_text_over_line(r'$\hat{\beta}_{IC0}=%f$'%ci.loc[predictor, 0],
                        slope=ci.loc[predictor, 0],
                        position=[x_bins[0], iv_l[0]], va='top')
    plot_text_over_line(r'$\hat{\beta}_{IC1}=%f$'%ci.loc[predictor, 1],
                        slope=ci.loc[predictor, 1],
                        position=[x_bins[0], iv_u[0]], va='bottom')

    y_fit_plot  = (estimate.params[predictor] * x_bins + estimate.params['const'] +
                   sum([estimate.params[c]*df[c].median() for c in ci.index
                        if c not in {predictor, 'const'}]))
    ax.plot(x_bins, y_fit_plot, reg_color, label=eq_label)
    ax.legend(loc="best", prop={'size': 10})

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout(pad=1.0)

    if fig_dir is not None:
        fig.savefig(op.join(fig_dir, '%s_corr_%s_%s%s.png' %
                            (predictor, outcome, '_'.join(covariables), file_suffix)))

    if report_dir is not None:
        summary_fn = op.join(report_dir, '%s_corr_%s_%s%s.txt' %
                             (predictor, outcome, '_'.join(covariables), file_suffix))
        with open(summary_fn, 'w') as fout:
            fout.write(str(estimate.summary()))

    b = estimate.params.rename('b')
    Beta = pd.Series({c:v*(df[c].std() if c !='const' else 1)
                      for c,v in estimate.params.items()},
                     name='Beta')
    t = estimate.tvalues.rename('t')
    p = estimate.pvalues.rename('p')
    stats_df = pd.concat((b, Beta, t, p), axis=1)
    stats_df.index.name = 'Variable'

    return estimate, stats_df, fig

def plot_text_over_line(text, slope, position, va='bottom'):
    angle = np.rad2deg(np.arctan2(slope, 1))
    plt.text(position[0], position[1], text, ha='left', va=va,
             transform_rotates_text=True, rotation=angle, rotation_mode='anchor')

def format_pvalue(pvalue, show_p=True, latex=False):
    if plt.rcParams['text.usetex']:
        c_less = ['<', r'\textless'][latex]
        bold_tag = '\\bfseries '
    else:
        c_less = ['<', '<'][latex]
        bold_tag = ''
    if pvalue < 0.001:
        return bold_tag + ['', 'p'][show_p] + c_less + '0.001'
    elif pvalue < 0.01:
        return bold_tag + ['', 'p'][show_p] + c_less + '0.01'
    elif pvalue < 0.05:
        return bold_tag + ['', 'p='][show_p] + '%1.2f' % pvalue
    else:
        return ['', 'p='][show_p] + '%1.2f' % pvalue

from functools import partial

def mediation(data, predictor, mediators, outcome, output_dir=None, covariates=None):

    if isinstance(mediators, str):
        mediators = [mediators]
    tag = '%s_mediation_%s_%s' % (predictor, '-'.join(mediators), outcome)
    print(tag)
    try:
        med = mediation_analysis(data=data, x=predictor, m=mediators, y=outcome,
                                 alpha=0.05,
                                 seed=42, covar=covariates).set_index('path')
    except:
        print('Error mediation')
        raise

    med['pval'] = med['pval']
    med['pval_stars'] = med['pval'].apply(partial(pval_to_stars, mc_corr=1))

    if output_dir is not None:
        f = graphviz.Digraph('Med', filename=op.join(output_dir, tag + '.gv'))
        f.attr(rankdir='LR', size='8,5')

        f.attr('node', shape='box')
        f.node(predictor)
        for mediator in mediators:
            f.node(mediator)
        f.node(outcome)

        pat = '%s=%1.2g%s(%1.2g)'
        for mediator in mediators:
            med_pred = med.loc['%s ~ X' % mediator]
            f.edge(predictor, mediator, label=pat % ('a', med_pred['coef'],
                                                     med_pred['pval_stars'], med_pred['se']))
            med_out = med.loc['Y ~ %s' % mediator]
            f.edge(mediator, outcome, label=pat % ('b', med_out['coef'],
                                                   med_out['pval_stars'], med_out['se']))
        pred_out = med.loc['Direct']
        f.edge(predictor, outcome, label=pat % ('c\'', pred_out['coef'],
                                                pred_out['pval_stars'], pred_out['se']))

        f.format = 'svg'
        f.render(directory=output_dir)

    med.loc['Indirect', 'coef_pct'] = ( med.loc['Indirect', 'coef'] /
                                        med.loc['Total', 'coef'] ) * 100
    med.loc['Direct', 'coef_pct'] = ( med.loc["Direct", 'coef'] /
                                      med.loc['Total', 'coef'] ) * 100

    #print(med.drop(columns=['pval_stars']))
    #print()

    return med

def format_factor(f):
    if f < 0.01:
        return '%1.4f' % f
    else:
        return '%1.3f' % f

def pval_to_stars(p, mc_corr=1):
    p = p / mc_corr
    p_stars = ''
    if p <= 0.05:
        p_stars = '*'
    if p <= 0.01:
        p_stars = '**'
    if p <= 0.001:
        p_stars = '***'
    return p_stars
