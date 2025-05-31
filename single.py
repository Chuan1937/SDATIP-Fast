import numpy as np
import plotresult
import plotresult_graduate
from Waveform import Waveform


def solutionset(name, data, outputdir):
    a = Waveform('%s' % (name))
    a.importdata(data, 0.01)
    a.analyzedata()
    a.interpolate(1)
    a.denseunique()
    a.denselong(5 / 2, 200)
    a.extremearr()
    a.densebin()
    b = a.constructstate()
    [c, c_num] = b.markovmatrix()
    d = b.ampprobcalculate()


    main_data_to_save = {
        'transitionmatrix': np.array(b.matrix),
        'ampprob': np.array(b.ampprob_up).astype('float64'),
        'Apeak': b.Apeak,
        'samplelength': b.samplength,
        'eigvalue': b.eigvalue,
        'bigeig': b.bigeig,
        'threshold': a.threshold
    }

    main_filepath = f'{outputdir}{a.name}.npz'
    np.savez_compressed(main_filepath, **main_data_to_save, allow_pickle=True)
    for i in range(0, c_num):
        b.estimation(i)
        timeprob_filepath = f'{outputdir}{a.name}_timeprob_{i}.npz'
        np.savez_compressed(timeprob_filepath, timeprob=c[i])

        plotresult.plotprob(a, b, i, a.name, outputdir)
        plotresult_graduate.plotprob(a, b, i, a.name, outputdir)
        txt_filepath = f'{outputdir}{name}.txt'
        with open(txt_filepath, "a") as f:
            line = (
                f'{name} solution id:{i} '
                f'arrivaltime:{b.arrivalestimate:.3f} '
                f'overall up:{float(np.sum(c[i] * d)):.5f} '
                f'up:{b.polarityup:.3f} '
                f'down:{b.polaritydown:.3f} '
                f'unknown:{b.polarityunknown:.3f}\n'
            )
            f.writelines(line)