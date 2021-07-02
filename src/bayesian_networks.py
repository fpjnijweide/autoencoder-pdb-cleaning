import pyAgrum as gum
import scipy.stats

from .probabilities import generate_samplespace


def make_bn(BN_size, sampling_density):
    bn = gum.BayesNet("Quasi-Continuous")
    new_nodes = []
    # a = bn.add(gum.LabelizedVariable("A", "A binary variable", 2))
    # bn.cpt(a)[:] = [0.4, 0.6]
    a = bn.add(gum.RangeVariable("A", "A range variable", 0, sampling_density - 1))
    # bn.addArc(a, b)
    # first = generate_samplespace(scipy.stats.truncnorm(-10, 3), -10, 3, sampling_density)
    # second = generate_samplespace(scipy.stats.truncnorm(-2, 6), -2, 6, sampling_density)
    gauss = generate_samplespace(scipy.stats.norm(-2, 2), -2, 2, sampling_density)
    # bn.cpt(b)[{'A': 0}] = first
    # bn.cpt(b)[{'A': 1}] = second
    bn.cpt(a)[:] = gauss
    new_nodes.append(a)

    if BN_size > 1:
        b = bn.add(gum.RangeVariable("B", "Another quasi continuous variable", 0, sampling_density - 1))
        bn.addArc(a,b)
        l = []
        for i in range(sampling_density):
            # the size and the parameter of gamma depends on the parent value
            k = (i * 30.0) / sampling_density
            l.append(generate_samplespace(scipy.stats.gamma(k + 1), 4, 5 + k, sampling_density))
        bn.cpt(b)[:] = l
        new_nodes.append(b)
    if BN_size > 2:
        c = bn.add(gum.RangeVariable("C", "Another quasi continuous variable", 0, sampling_density - 1))
        bn.addArc(b, c)
        l = []
        for i in range(sampling_density):
            # the size and the parameter of gamma depends on the parent value
            k = (i * 30.0) / sampling_density
            l.append(generate_samplespace(scipy.stats.gamma(k + 1), 4, 5 + k, sampling_density))
        bn.cpt(c)[:] = l
        new_nodes.append(c)

        for d in range(BN_size - 3):
            # new variable
            new_nodes.append(
                bn.add(gum.RangeVariable("D" + str(d), "Another quasi continuous variable", 0, sampling_density - 1)))
            l = []
            bn.addArc(new_nodes[-2], new_nodes[-1])
            for i in range(sampling_density):
                # the size and the parameter of gamma depends on the parent value
                k = (i * 30.0) / sampling_density
                l.append(generate_samplespace(scipy.stats.gamma(k + 1), 4, 5 + k, sampling_density))
            bn.cpt(new_nodes[-1])[:] = l

    return bn
