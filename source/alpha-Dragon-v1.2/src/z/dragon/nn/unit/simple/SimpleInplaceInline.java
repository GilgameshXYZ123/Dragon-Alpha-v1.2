/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple;

import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public abstract class SimpleInplaceInline<T extends SimpleInplaceUnit> extends SimpleCore<T> 
{
    public SimpleInplaceInline(T unit) { super(unit); }

   //<editor-fold defaultstate="collapsed" desc="functions">
    public final boolean inplace() { return ut.inplace; }
    
    protected abstract Tensor __forward__(Engine eg, Tensor X, boolean inplace);
    
    @Override 
    protected final Tensor __forward__(Engine eg, Tensor X) {
        return __forward__(eg, X, ut.inplace).modify(ut.inplace);
    }
    //</editor-fold>
}
