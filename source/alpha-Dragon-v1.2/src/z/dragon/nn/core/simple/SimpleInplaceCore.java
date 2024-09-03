/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public abstract class SimpleInplaceCore<T extends SimpleUnit> extends SimpleCore<T> {
    private final boolean inplace;
    
    protected SimpleInplaceCore(T unit, boolean inplace) {
        super(unit);
        this.inplace = inplace;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final boolean inplace() { return inplace; }
    
    protected abstract Tensor __forward__(Engine eg, Tensor X, boolean inplace);
    
    @Override 
    protected final Tensor __forward__(Engine eg, Tensor X) {
        return __forward__(eg, X, inplace).modify(inplace);
    }
    //</editor-fold>
}
