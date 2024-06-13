/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.furcation;

import java.util.Collection;
import java.util.HashSet;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.core.Trace;
import z.dragon.nn.core.UnitCore;
import z.dragon.nn.unit.furcation.Furcation;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public abstract class FurcationCore <T extends Furcation> extends UnitCore<T> 
{
    //<editor-fold defaultstate="collapsed" desc="member-parameters">
    transient private UnitCoreMap<Object>[] arcs;//solve the topology
    transient private final UnitCoreSet nexts = new UnitCoreSet();
    
    transient private int output_tensor_num = -1;
    
    transient private Tensor   X, deltaX;//input, input.grads
    transient private Tensor[] Y, deltaY;//output, output.grads
    transient private boolean last_need_grads;
    
    transient private int   X_mod_count = -1;
    transient private int[] Y_mod_count; 
    //</editor-fold>
    
    public FurcationCore(T unit) { super(unit); }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int output_tensor_num() { return output_tensor_num; }
    
    public final Tensor   X() { return X; }
    public final Tensor[] Y() { return Y; }
    
    public final Tensor   deltaX() { return deltaX; }
    public final Tensor[] deltaY() { return deltaY; }
    
    public Tensor holdX() { return (X.check()? X.hold(X_mod_count, name() + ".X") : X); }
    public Tensor holdY(int index) { 
        Tensor Yi = Y[index];
        return (Yi.check()? Y[index].hold(Y_mod_count[index], name() +  ".Y[" + index + "]") : Yi); 
    }
    
    public boolean isHoldX() { return X.is_hold(X_mod_count); }
    public boolean isHoldY(int index) { return Y[index].is_hold(Y_mod_count[index]); } 
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override public Collection<UnitCore<?>> next() { return nexts; }

    @Override
    public void variables(TensorSet set) {
        set.add(X, deltaX);
        set.add(Y); set.add(deltaY);
    }
    
    @Override
    public void gc() {
        if(X != null) { X.delete(); X = null; }
        if(Y != null) { Tensor.delete(Y); Y = null;} 
        if(deltaX != null) { deltaX.delete(); deltaX = null; }
        if(deltaY != null) { Tensor.delete(deltaY); deltaY = null; }
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
    protected abstract Tensor[] __forward__(Engine eg, Tensor X);
    
    @Override
    public Tensor[] forward(Tensor... input) {
        nexts.clear(); arcs = null; //all one-off UnitCores reference has been cleared
        
        //receive from last: I recevice your output, and it's my 0th input
        X = input[0]; Trace trace = X.trace(); if(trace != null) X.trace().callback(this, 0);
      
        X_mod_count = X.mod_count();//save: X.mod_count
        
        Y = __forward__(X.engine(), X.c());//wait until X is cauculated
      
        Y_mod_count = new int[output_tensor_num = Y.length];//save Y.mod_count
        for(int i=0; i<output_tensor_num; i++) Y_mod_count[i] = Y[i].mod_count();
        
        arcs = new UnitCoreMap[output_tensor_num];//one graph mapped to one output, used to collect gradient
        
        //send trace to the next Unit: send my ith output to you
        last_need_grads = X.need_grad();
        if(trace != null) last_need_grads = (last_need_grads || trace.need_grads());
        boolean need_grads = (last_need_grads || ut.need_grads());
        for(int i=0; i<Y.length; i++) Y[i].setTrace(this, i, need_grads);
        
        return Y;
    }

    @Override
    protected void traceBack(UnitCore<?> next, int out_index, int next_in_index) {
        nexts.add(next);
        
        if(arcs[out_index] == null) arcs[out_index] = new UnitCoreMap<>(2);
        UnitCoreMap<Object> arc = arcs[out_index];
        
        Object value = arc.get(next);
        if(value == null) arc.put(next, next_in_index);
        else if(value instanceof Integer) {
            HashSet<Integer> indexes = new HashSet<>(4);
            indexes.add((Integer)value);
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
        boolean allNull = true;
        deltaY = new Tensor[arcs.length];
        
        for(int i=0; i<arcs.length; i++) {
            if(arcs[i] == null) continue;
            deltaY[i] = this.aggregateGradient(arcs[i]); arcs[i] = null;
            //if deltaY != null: allNull = (allNull && false) = false
            allNull = allNull && (deltaY[i] == null);//at least one grad != null
        }
        
        arcs = null;
        return (allNull? null : deltaY);
    }

    protected abstract Tensor __backward__(Engine eg, Tensor[] deltaY,//compute deltaX
            boolean grad_inplace, boolean[] grad_inplaces, 
            boolean backward_grads);
    
    @Override
    public Tensor[] backward(Tensor... gradient) {//no gradient, no backward
        if(gradient == null) { deltaY = null; deltaX = null; return null; }
        
        deltaY = gradient;//deltaYarr[0] = gradient[0] = deltaYarr[0]
        
        if(before_backward != null) before_backward.callback(this);
        
        //backward_grads: when last unit need grads, and backward_grads_switch is on
        boolean[] grad_inplaces = new boolean[output_tensor_num];
        boolean grad_inplace = false;
        for(int i=0; i<output_tensor_num; i++) {
            grad_inplaces[i] = (!Y[i].need_grad());
            grad_inplace = (grad_inplace || grad_inplaces[i]);
        }
        boolean backward_grads = (ut.backward_grads() && last_need_grads);
        if(deltaY.length != output_tensor_num) throw new IllegalArgumentException(String.format(
                "FurcationCore: gradients.length != output_tensor_num", 
                deltaY.length, output_tensor_num));
        deltaX = __backward__(deltaY[0].engine(), deltaY, 
                grad_inplace, grad_inplaces, 
                backward_grads);
        
        if(X.need_grad()) { X.grad(deltaX); }//collect gradient for X
        
        if(after_backward != null) after_backward.callback(this);
        
        return (backward_grads? new Tensor[]{ deltaX } : null);
    }

    @Override
    public Tensor gradient(int index) {
        if(index != 0) throw new IllegalArgumentException("tensor index out of range");
        return deltaX;
    }
    //</editor-fold>
}
