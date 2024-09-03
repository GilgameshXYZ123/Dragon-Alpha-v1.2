/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.tensor;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.simple.SimpleInplaceCore;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreTrim<T extends SimpleUnit> extends SimpleInplaceCore<T> {
    protected int[] t0;//trimming on the start
    protected int[] t1;//trimming on the end
    
    public CoreTrim(T unit, boolean inplace, int[] t0, int[] t1) {
        super(unit, inplace);
        this.t0 = t0;
        this.t1 = t1;
    }
    
    public final int[] t0() { return t0; }
    public final int[] t1() { return t1; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        return eg.trim(inplace, X, t0, t1);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads?
                eg.pad(grad_inplace, deltaY, t0, t1) :
                null);
    }
    //</editor-fold>
}
