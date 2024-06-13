/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.reducer;

import java.util.Collection;
import java.util.HashSet;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.core.Trace;
import z.dragon.nn.core.UnitCore;
import z.dragon.nn.unit.reducer.Reducer;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
@SuppressWarnings("unchecked")
public abstract class ReducerCore<T extends Reducer> extends UnitCore<T>
{
    //<editor-fold defaultstate="collapsed" desc="member-parameters">
    transient private final UnitCoreMap<Object> arc = new UnitCoreMap<>();//solve the topology
    
    transient private int input_tensor_num = -1;
    
    transient private Tensor[] X, deltaX;//input, input.grads
    transient private Tensor   Y, deltaY;//output, output.grads
    transient private boolean[] last_need_grads;
    transient private boolean LAST_need_grads;
    
    transient private int[] X_mod_count;
    transient private int   Y_mod_count;
    //</editor-fold>
    
    public ReducerCore(T unit) { super(unit); }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final int input_tensor_num() { return input_tensor_num; }
    
    public final Tensor[] X() { return X; }
    public final Tensor   Y() { return Y; }
    
    public final Tensor[] deltaX() { return deltaX; }
    public final Tensor   deltaY()  {return deltaY; }
    
    public final Tensor holdY() { return (Y.check() ? Y.hold(Y_mod_count, name() + ".Y") : Y); }
    public final Tensor holdX(int index) {  
        Tensor Xi = X[index];
        return (Xi.check()?  Xi.hold(X_mod_count[index], name() + ".X[" + index + "]") : Xi);  
    }
    
    public final boolean isHoldX(int index) { return X[index].is_hold(X_mod_count[index]); }
    public final boolean isHoldY(int index) { return Y.is_hold(Y_mod_count); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override public Collection<UnitCore<?>> next() { return arc.keySet(); }

    @Override
    public void variables(TensorSet set) {
        set.add(X); set.add(deltaX);
        set.add(Y, deltaY);
    }
    
    @Override
    public void gc() {
        if(X != null) { Tensor.delete(X); X = null; }
        if(Y != null) { Y.delete(); Y = null; }
        if(deltaX != null) { Tensor.delete(deltaX); deltaX = null; }
        if(deltaY != null) { deltaY.delete(); deltaY = null; }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
    protected abstract Tensor __forward__(Engine eg, Tensor[] X);
    
    @Override
    public Tensor[] forward(Tensor... input) {
        arc.clear();//all one-off UnitCore references have been cleared
       
        X = input; input_tensor_num = X.length;
        Trace[] trace = new Trace[input_tensor_num];
        X_mod_count = new int[input_tensor_num];
        for(int i=0; i< X.length; i++) {
            //receive from last: I recevice your output, and it's my ith input
            if((trace[i] = X[i].trace()) != null) trace[i].callback(this, i);
            X_mod_count[i] = X[i].mod_count();//save: X.mod_count
        }
        
        Tensor.sync(X);//wait until X is cauculated
        Y = __forward__(X[0].engine(), X);

        Y_mod_count = Y.mod_count();//save: Y.mod_count
        
        //send trace to the next: I will send my 0th output to you
        last_need_grads = new boolean[input_tensor_num];//mapping: X[i]
        LAST_need_grads = false;//mapping: X[0 : n]
        for(int i=0; i<X.length; i++) {
            last_need_grads[i] = X[i].need_grad();
            if(trace[i] != null) last_need_grads[i] = (last_need_grads[i] || trace[i].need_grads());
            LAST_need_grads = (LAST_need_grads || last_need_grads[i]);
        }
        
        boolean need_grads = (ut.need_grads() || LAST_need_grads);
        Y.setTrace(this, 0, need_grads);
        return new Tensor[]{ Y };
    }

    @Override
    protected void traceBack(UnitCore<?> next, int out_index, int next_in_index) {
        Object value = arc.get(next);
        if(value == null) arc.put(next, next_in_index);
        else if(value instanceof Integer) {
            HashSet<Integer> indexes = new HashSet<>(4);
            indexes.add((Integer) value);
            indexes.add(next_in_index);
            arc.put(next, indexes);
        }
        else {//value instance of HashSet<Integer>
            HashSet<Integer> indexes = (HashSet<Integer>) value;
            indexes.add(next_in_index);
        }
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
    @Override
    public Tensor[] collectGradientFromNext() {
        deltaY = this.aggregateGradient(arc);
        arc.clear();
        return (deltaY == null? null : new Tensor[]{ deltaY });
    }

    //backward_grads = false: backward_grads_inde.length = 0
    protected abstract Tensor[] __backward__(Engine eg, Tensor deltaY, 
            int input_tensor_num, boolean grad_inplace, 
            boolean backward_grads, boolean[] last_need_grads);
    
    protected int[] backward_grads_index(boolean[] last_need_grads) {
        int count = 0;
        for(int i=0; i<input_tensor_num; i++) if(last_need_grads[i]) count++;
        int[] backward_grads_index = new int[count];
        for(int idx = 0, i=0; i<input_tensor_num; i++) 
            if(last_need_grads[i]) backward_grads_index[idx++] = i;
        return backward_grads_index;
    }
    
    @Override
    public Tensor[] backward(Tensor... gradient) {//no gradient, no backward
        if(gradient == null) { deltaY = null; deltaX = null; return null; }
        
        deltaY = gradient[0];//deltaYarr[0] = gradient[0] = deltaYarr[0]
        
        if(before_backward != null) before_backward.callback(this);
        
        //backward_grads: when last unit need grads, and backward_grads_switch is on
        boolean grad_inplace = (!Y.need_grad());
        boolean backward_grads = (ut.backward_grads() && LAST_need_grads);
        deltaX = __backward__(deltaY.engine(), deltaY.c(), 
                input_tensor_num, grad_inplace, 
                backward_grads, last_need_grads);
        
        if(deltaX != null)//collect gradient for X
            for(int i=0; i<X.length; i++) if(X[i].need_grad()) X[i].grad(deltaX[i]); 
   
        if(after_backward != null) after_backward.callback(this);
        
        return (backward_grads? deltaX : null);
    }

    @Override
    public Tensor gradient(int index) {
        if(index > deltaX.length || index < 0) throw new IllegalArgumentException("tensor index out of range");
        return (deltaX == null? null : deltaX[index]);
    }
    //</editor-fold>
}
