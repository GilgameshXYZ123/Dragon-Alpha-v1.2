/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.module;

import java.util.Collection;
import java.util.HashSet;
import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.core.Trace;
import z.dragon.nn.core.UnitCore;
import z.dragon.nn.unit.complex.Module;

@SuppressWarnings("unchecked")
public class CoreGraphTail<T extends Module> extends UnitCore<T> {
    //<editor-fold defaultstate="collapsed" desc="member-parameters">
    transient private UnitCoreMap<Object>[] arcs;//Solve the topology,{ int or HashSet<Integer> }
    transient UnitCoreSet nexts = new UnitCoreSet();

    transient private final CoreModule core_module;
    transient private Tensor[] X;//input
    transient private Tensor[] deltaX;//input.gradient
    //</editor-fold>

    protected CoreGraphTail(T unit, CoreModule core_module) { 
        super(unit);
        this.core_module = core_module;
    }

    //<editor-fold defaultstate="collapsed" desc="function: others">
    @Override public Collection<UnitCore<?>> next() { return nexts; }
    
    @Override 
    public void variables(TensorSet set) { 
        set.add(X); set.add(deltaX); 
    }
    
    @Override 
    public void gc() { 
        if (X != null) { Tensor.delete(X); X = null; }
        if (deltaX != null) { Tensor.delete(deltaX); deltaX = null; } 
        nexts.clear(); 
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
    @Override
    public Tensor[] forward(Tensor... input) {
        X = input;//all one-off UnitCore references has been cleared
        nexts.clear(); arcs = new UnitCoreMap[input.length];
        
        for(int i=0; i <X.length; i++) {
            Trace trace = X[i].trace();
            trace.callback(this, i);//ends.next = tail
            boolean need_grads = (X[i].need_grad() || trace.need_grads());
            X[i].setTrace(core_module, i, need_grads);//the output sender is module
        }
        return X;
    }

    @Override
    protected synchronized void traceBack(UnitCore next, int out_index, int next_in_index) {
        nexts.add(next);//module.nexts = nexts

        if (arcs[out_index] == null) arcs[out_index] = new UnitCoreMap<>(2);
        UnitCoreMap<Object> graph = arcs[out_index];
        
        Object value = graph.get(next);
        if (value == null) graph.put(next, next_in_index);
        else if (value instanceof Integer) {
            HashSet<Integer> indexes = new HashSet<>(4);
            indexes.add((Integer) value);
            indexes.add(next_in_index);
            graph.put(next, indexes);
        }
        else {//tensor[out_index].used_size >= 2
            HashSet<Integer> indexes = (HashSet<Integer>) value;
            indexes.add(next_in_index);
        }
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
    @Override
    public Tensor[] collectGradientFromNext() {
        if(!ut.backward_grads()) return null;//backward_grads = false
        
        boolean allNull = true;
        deltaX = new Tensor[arcs.length];
        
        for (int i = 0; i < arcs.length; i++) {
            if (arcs[i] == null) continue;
            deltaX[i] = aggregateGradient(arcs[i]); arcs[i] = null;
            //if deltaX[i] != null: allNull = (allNull && false) = false
            allNull = allNull && (deltaX[i] == null);//at least one grad != null
        }
        arcs = null;
        
        //if all gradients from the next Units are all null:
        //means all next scales doesn't do backward prop, so the Modile Tail needn't do either
        return (allNull? null : deltaX);
    }

    @Override//Make Sure：deltaX.size == input.size, and the two are aligned 
    public Tensor[] backward(Tensor... gradient) { 
        return deltaX = gradient; 
    }
    
    @Override 
    public Tensor gradient(int index) {
        if (index > deltaX.length || index < 0) throw new IllegalArgumentException("tensor index out of range");
        return (deltaX == null ? null : deltaX[index]);
    }
    //</editor-fold>
}
