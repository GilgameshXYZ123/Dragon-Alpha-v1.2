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
public class CoreTranspose<T extends SimpleUnit> extends SimpleInplaceCore<T> {
    protected int idx1;//dim Idx1
    protected int idx2;//dim Idx2
    
    public CoreTranspose(T unit, boolean inplace, int dimIdx1, int dimIdx2) {
        super(unit,  inplace);
        this.idx1 = dimIdx1;
        this.idx2 = dimIdx2;
    }
    
    public final int dimIdx1() { return idx1; }
    public final int dimIdx2() { return idx2; }
    public final int[] dimIdx() { return new int[]{ idx1, idx2 }; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
        return eg.transpose(inplace, X, idx1, idx2);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY,
            boolean grad_inplace, boolean backward_grads) {
        return (backward_grads? 
                eg.transpose(grad_inplace, deltaY, idx1, idx2):
                null);
    }
    //</editor-fold>
}
