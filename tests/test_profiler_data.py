"""Owner's desktop profiler contract: measured, finite, bounded and internally consistent."""
import copy

import pytest

from wonhee.profiler_data import ProfilerData


def data_fixture():
    def stats(a,b):
        return {'n':2,'median_us':(a+b)/2,'mean_us':(a+b)/2,'p95_us':b,'p99_us':b,'min_us':a,'max_us':b}
    return {'seed':24026,'summary':{'stdio':stats(20.,22.),'http':stats(180.,200.),
            'http_over_stdio_median_rtt':190/21},'histogram':{'edges_us':[10.,100.,1000.],'stdio':[2,0],'http':[0,2]},
            'sample_stride':1,'pairs':[{'pair_id':'p0','payload_bytes':500,'stdio_us':20.,'http_us':180.},
                                     {'pair_id':'p1','payload_bytes':1000,'stdio_us':22.,'http_us':200.}],
            'paragraph_previews':['random words'],'sample_count_per_transport':2,'paragraph_count':2,
            'scope':'single-worker local JSON echo','animation':'time-scaled dataset replay'}


def test_valid_profiler_data_retains_values():
    assert ProfilerData.model_validate(data_fixture()).model_dump()==data_fixture()


@pytest.mark.parametrize('change',[
    lambda d:d.update(summary=None),lambda d:d.update(pairs='bogus'),
    lambda d:d['pairs'][0].update(stdio_us=float('nan')),
    lambda d:d['pairs'][0].update(http_us=float('inf')),
    lambda d:d['summary'].update(http_over_stdio_median_rtt=999.),
    lambda d:d['histogram'].update(stdio=[1,0]),
    lambda d:d['histogram'].update(edges_us=[10000.,20000.],stdio=[2],http=[2]),
    lambda d:d.update(sample_stride=2),
    lambda d:d['histogram'].update(edges_us=[100.,10.,1000.]),
    lambda d:d.update(scope=[]),lambda d:d['pairs'].append(copy.deepcopy(d['pairs'][0])),
])
def test_invalid_profiler_data_is_rejected(change):
    data=data_fixture();change(data)
    with pytest.raises(ValueError):
        ProfilerData.model_validate(data)
