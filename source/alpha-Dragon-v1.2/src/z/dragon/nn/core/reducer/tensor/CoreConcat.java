/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.reducer.tensor;

import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.core.reducer.ReducerCore;
import z.dragon.nn.unit.reducer.Reducer;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreConcat<T extends Reducer> extends ReducerCore<T> {
    protected int dimIdx;
    
    transient protected int[] section;
    
    public CoreConcat(T unit, int dimIdx) {
        super(unit);
        this.dimIdx = dimIdx;
    }
    
    public final int dimIdx() { return dimIdx; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor[] X) {
        Tensor Y = eg.concat(dimIdx, X);
        section = new int[X.length];
        for(int i=0; i<X.length; i++) section[i] = X[i].dim(dimIdx);
        return Y;
    }

    @Override
    protected Tensor[] __backward__(Engine eg, Tensor deltaY, 
            int input_tensor_num, boolean grad_inplace, 
            boolean backward_grads, boolean[] last_need_grads)
    {
        if(!backward_grads) return null;//grads_num = 0;
        
        Tensor[] deltaX = eg.split(deltaY, last_need_grads, dimIdx, section);
        
        if(grad_inplace) {//when deltaX[] are found, deltaY is not needed
            int grad_num = last_need_grads.length;
            int[] backward_grads_index = backward_grads_index(last_need_grads);
            CountGc gc = new CountGc(grad_num, deltaY);
            for(int i=0; i< grad_num; i++) {
                int idx = backward_grads_index[i];
                deltaX[idx].dual(()-> { gc.countDown(); });
            }
        }
        
        return deltaX;
    }
    //</editor-fold>
}
