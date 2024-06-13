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
public class CorePad<T extends SimpleUnit> extends SimpleInplaceCore<T>
{
    protected int[] p0;//padding on the start
    protected int[] p1;//padding on the end
    
    public CorePad(T unit, boolean inplace, int[] p0, int[] p1) {
        super(unit, inplace);
        this.p0 = p0;
        this.p1 = p1;
    }
    
    public final int[] p0() { return p0; }
    public final int[] p1() { return p1; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        return eg.pad(inplace, X, p0, p1);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads?
                eg.trim(grad_inplace, deltaY, p0, p1) :
                null);
    }
    //</editor-fold>
}
