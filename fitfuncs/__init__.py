import fitfuncs.TrackerAngRes_FloatPower as _TrackerAngRes_FloatPower
import fitfuncs.TrackerAngRes_NoQuad as _TrackerAngRes_NoQuad
import fitfuncs.TrackerAngRes_QuadAdd as _TrackerAngRes_QuadAdd
import fitfuncs.Linear as _Linear
import fitfuncs.Quadratic as _Quadratic
import fitfuncs.Quartic as _Quartic
import fitfuncs.Proportional as _Proportional
import fitfuncs.Power as _Power
import fitfuncs.PowerPlusConst as _PowerPlusConst

funcs = {
    'TrackerAngRes_FloatPower': _TrackerAngRes_FloatPower,
    'TrackerAngRes_NoQuad'    : _TrackerAngRes_NoQuad,
    'TrackerAngRes_QuadAdd'   : _TrackerAngRes_QuadAdd,
    'Linear'                  : _Linear,
    'Quadratic'               : _Quadratic,
    'Quartic'                 : _Quartic,
    'Proportional'            : _Proportional,
    'Power'                   : _Power,
    'PowerPlusConst'          : _PowerPlusConst
}

def get_func(name):
    return funcs[name]
