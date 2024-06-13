/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.blas;

import z.dragon.common.state.State;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter;
import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 */
public class DepthWiseDeconv3D extends SimpleUnit
{
    protected int IC;
    protected int FH, FW;
    protected int sh, sw;
    protected int ph, pw;
    protected int OH, OW;
    protected boolean biased;

    @Override
    protected SimpleCore<?> create_unit_core() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void append(String pre, StringBuilder sb) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void params(Parameter.ParamSet set) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void param_map(Parameter.ParamMap<String> map) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void update_state(State dic, boolean partial) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    protected void __init__(Engine eg) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void state(State dic) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
