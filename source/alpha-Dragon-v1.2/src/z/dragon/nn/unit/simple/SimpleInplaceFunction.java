/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple;

import z.dragon.common.state.State;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter;

/**
 * @author Gilgamesh
 */
public abstract class SimpleInplaceFunction extends SimpleInplaceUnit
{
    private static final long serialVersionUID = 1L;
    
    protected SimpleInplaceFunction(boolean inplace) { super(inplace); }
    
    @Override protected void __init__(Engine eg) {}
    @Override public void params(Parameter.ParamSet set) {}
    @Override public void param_map(Parameter.ParamMap<String> map) {}
    @Override public void state(State dic) {}
    @Override public void update_state(State dic, boolean partial) {}
}
