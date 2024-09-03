/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.function.BiConsumer;
import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorList;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.unit.Unit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
@SuppressWarnings("unchecked")
public abstract class UnitCore<T extends Unit> implements BackwardHookable {
    //<editor-fold defaultstate="collapsed" desc="static class: UnitCoreMap">
    public static class UnitCoreMap<V> extends HashMap<UnitCore<?>, V> 
    {
        private static final long serialVersionUID = 141558956307138L;
        
        public UnitCoreMap() {super();}
        public UnitCoreMap(int init_capacity) { super(init_capacity); }

        @Override
        public final V put(UnitCore<?> node, V value) {
            return (node == null ? null : super.put(node, value));
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="static class: UnitCoreSet">
    public static class UnitCoreSet extends HashSet<UnitCore<?>>
    {
        private static final long serialVersionUID = 1L;
        
        public UnitCoreSet() { super(); }
        public UnitCoreSet(int init_capacity) { super(init_capacity); }

        @Override
        public final boolean add(UnitCore<?> node) {
            return (node == null ? false :  super.add(node));
        }
        
        public final boolean add(UnitCore<?>... nodes) {
            if(nodes == null || nodes.length == 0) return false;
            boolean result = true;
            for(UnitCore<?> node : nodes) {
                result &= (node == null ? false : super.add(node));
            }
            return result;
        }
    }
    //</editor-fold>

    protected transient T ut;
    
    public UnitCore(T unit) { this.ut = unit; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    public final T unit() { return ut; }
    public final String name() { return ut.name() + "<Core" + ut.index_of_core(this) + "," + this.getClass().getName() + ">"; }
    
    public final boolean is_complex() { return ut.is_complex(); }
    public final boolean need_grads() { return ut.need_grads(); }
    public final boolean backward_grads() { return ut.backward_grads(); }
    
    public abstract Collection<UnitCore<?>> next();//the references for next nodes will be cleared after backward
    public abstract void gc();
    public abstract void variables(TensorSet set);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    public abstract Tensor[] forward(Tensor... X);
    protected abstract void traceBack(UnitCore<?> next, int out_index, int next_in_index);
    
    public abstract Tensor[] collectGradientFromNext();
    public abstract Tensor[] backward(Tensor... gradient);
    public abstract Tensor gradient(int index);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: gradient-aggregation">
    //<editor-fold defaultstate="collapsed" desc="class: GradientAggregator">
    public static class GradientAggregator implements BiConsumer<UnitCore<?>, Object> {
        private final TensorList grads = new TensorList(4);
        public final TensorList grads() { return grads; }
        
        @Override
        public void accept(UnitCore<?> next, Object value) {
            if (value == null) return;
            if (value instanceof Integer) {//used as 1 input: 1 -> 1
                grads.add(next.gradient((int) value));
            }
            else {//used as multiple input: 1 -> m
                HashSet<Integer> indice = (HashSet<Integer>) value;
                for (int index : indice) grads.add(next.gradient(index));
            }
        }
    }
    //</editor-fold>
    private final transient GradientAggregator aggr = new GradientAggregator();
    
    protected final Tensor aggregateGradient(Map<UnitCore<?>, Object> arc) {
        //System.out.println("BP: " + ut.name() + ", " + this + ". " + this.hashCode());
        
        arc.forEach(aggr); 
        TensorList grads = aggr.grads;
        if (grads.isEmpty()) return null;
        
        Tensor deltaY; int gsize = grads.size();
        if (gsize == 1) { deltaY = grads.get(0); grads.clear(); }
        else if (gsize == 2) {
            Tensor.sync(grads);//wait all grads are cauculated
            deltaY = grads.get(0).engine().sum(true, grads);
            deltaY.carry(grads.get(1));
            deltaY.dual(()-> { grads.get(1).delete(); grads.clear(); });
        }
        else if (gsize == 3) {
            Tensor.sync(grads);//wait all grads are cauculated
            deltaY = grads.get(0).engine().sum(true, grads).c();
            deltaY.carry(grads.get(1)); deltaY.carry(grads.get(2));
            deltaY.dual(()-> { grads.get(1).delete(); grads.get(2).delete(); grads.clear(); });
        }
        else {//find the summary of gradients
            Tensor.sync(grads);//wait all grads are cauculated
            deltaY = grads.get(0).engine().sum(true, grads);
            for (Tensor grad : grads) deltaY.carry(grad);//gc for oneOff nodes, if grad.need_carry = true
            deltaY.dual(()-> {//when deltaY is cauculated, grad[1:n-1] are not in need
                Iterator<Tensor> iter = grads.iterator();
                for (iter.next(); iter.hasNext(); ) iter.next().delete();//exclude grad[0]
                grads.clear();
            });
        }
        return deltaY;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: hook-mechanism">
    transient protected Hook before_backward;
    transient protected Hook after_backward;
    
    @Override public UnitCore hook_before_backward(Hook hook) { before_backward = hook; return this; }
    @Override public UnitCore hook_after_backward(Hook hook) { after_backward = hook; return this; }
    //</editor-fold>
}
