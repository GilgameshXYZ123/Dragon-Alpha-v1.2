/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.furcation.tensor;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.furcation.FurcationCore;
import z.dragon.nn.unit.furcation.Furcation;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreSplit<T extends Furcation> extends FurcationCore<T> {
    protected int dimIdx;
    protected int[] section;
    
    public CoreSplit(T unit, int dimIdx, int... section) {
        super(unit);
        this.dimIdx = dimIdx;
        this.section = section;
    }
    
    public final int dimIdx() { return dimIdx; }
    public final int[] section() { return section; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor[] __forward__(Engine eg, Tensor X) {
        return eg.split(X, dimIdx, section);
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor[] deltaY, 
            boolean grad_inplace, boolean[] grad_inplaces, 
            boolean backward_grads) 
    {
        if(!backward_grads) return null;
        
        Tensor deltaX = eg.concat(dimIdx, deltaY);
        
        if(grad_inplace) deltaX.dual(()-> { 
            for(int i=0; i<grad_inplaces.length; i++)
                if(grad_inplaces[i]) deltaY[i].delete();
        });
        
        return deltaX;
    }
    //</editor-fold>
}
