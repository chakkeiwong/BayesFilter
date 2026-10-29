"""Explicit estimands for simpler-target posterior assessment."""
import tensorflow as tf


DIAGNOSTIC_REVISION=2
SHAPE_DIAGNOSTIC_PROFILE='moments_regions_shape_v3'
RARE_EVENT_DIAGNOSTIC_PROFILE='moments_regions_shape_v4'
LEGACY_DIAGNOSTIC_PROFILE='moments_regions_v2'
MIXTURE_CDF_CUTS=(-7.,-5.,-3.,-2.,0.,2.,3.,5.,7.)
SECOND_CDF_CUTS=(-2.,0.,2.)


def features(target,samples,*,profile=LEGACY_DIAGNOSTIC_PROFILE):
    if profile not in (LEGACY_DIAGNOSTIC_PROFILE,SHAPE_DIAGNOSTIC_PROFILE,RARE_EVENT_DIAGNOSTIC_PROFILE):
        raise ValueError('unknown posterior diagnostic profile')
    flat=tf.reshape(samples,[-1,target.parameter_dim])
    regions=target.region_features(flat)
    if target.name in ('mixture','warped_mixture'):
        # Responsibilities are useful mean diagnostics but their extreme
        # quantiles numerically saturate. Use the actual half-space indicator
        # for regional ESS/precision; retain physical-coordinate tail ESS.
        regions=regions[:,-1:]
    values=tf.concat((flat,flat**2,regions),axis=1)
    if profile in (SHAPE_DIAGNOSTIC_PROFILE,RARE_EVENT_DIAGNOSTIC_PROFILE) and target.name in ('mixture','warped_mixture'):
        first=flat[:,:1]
        second=flat[:,1:2] if target.name=='mixture' else flat[:,1:2]-.1*(first**2-26.)
        values=tf.concat((values,
            tf.cast(first<tf.constant([MIXTURE_CDF_CUTS],flat.dtype),flat.dtype),
            tf.cast(tf.abs(first)<2.,flat.dtype),
            tf.cast(second<tf.constant([SECOND_CDF_CUTS],flat.dtype),flat.dtype)),axis=1)
    return tf.reshape(values,tf.concat((tf.shape(samples)[:-1],[tf.shape(values)[-1]]),0))


def feature_names(target,*,profile=LEGACY_DIAGNOSTIC_PROFILE):
    count=int(features(target,tf.zeros([1,target.parameter_dim],tf.float64),profile=profile).shape[-1])
    d=target.parameter_dim
    extra=13 if profile in (SHAPE_DIAGNOSTIC_PROFILE,RARE_EVENT_DIAGNOSTIC_PROFILE) and target.name in ('mixture','warped_mixture') else 0
    names=[f'x{i}' for i in range(d)]+[f'x{i}_squared' for i in range(d)]+[
        f'region{i}' for i in range(count-2*d-extra)]
    if extra:
        names += [f'x0_lt_{v:g}' for v in MIXTURE_CDF_CUTS]+['valley_abs_x0_lt2']+[
            f'unwarped_x1_lt_{v:g}' for v in SECOND_CDF_CUTS]
    return tuple(names)
