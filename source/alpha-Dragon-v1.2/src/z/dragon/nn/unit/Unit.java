/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit;

import z.dragon.nn.core.UnitCore;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import z.dragon.common.state.State;
import z.dragon.common.state.State.StateReader;
import z.dragon.engine.Tensor;
import z.dragon.common.state.State.Stateful;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.core.BackwardHookable;
import z.dragon.nn.core.Hook;

/**
 *
 * @author Gilgamesh
 */
@SuppressWarnings("unchecked")
public abstract class Unit implements Stateful, StateReader, 
        GradientControllable, BackwardHookable,
        Serializable
{
    private static final long serialVersionUID = 56278124000000000L;
    
    //<editor-fold defaultstate="collapsed" desc="static class: UnitMap">
    public static class UnitMap<V> extends HashMap<Unit, V> 
    {
        private static final long serialVersionUID = 141558956307138L;
        
        public UnitMap() {super();}
        public UnitMap(int init_capacity) { super(init_capacity); }

        @Override
        public final V put(Unit unit, V value) {
            return (unit == null ? null : super.put(unit, value));
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="static class: UnitSet">
    public static class UnitSet extends HashSet<Unit> 
    {
        private static final long serialVersionUID = 1L;
        
        public UnitSet() { super(); }
        public UnitSet(int init_capacity) { super(init_capacity); }

        @Override
        public final boolean add(Unit unit) {
            return (unit == null ? false :  super.add(unit));
        }
        
        public final boolean add(Unit...units) {
            if(units == null || units.length == 0) return false;
            boolean result = true;
            for(Unit unit : units) {
                result &= (unit == null ? false : super.add(unit));
            }
            return result;
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="daemon exec">
    protected static final ThreadFactory daemonThreadFactory = (Runnable r) -> {
        Thread t = new Thread(r);
        t.setDaemon(true);
        return t;
    };
    protected static final ExecutorService exec = Executors.newFixedThreadPool(4, daemonThreadFactory); 
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: UnitCoreManager">
    public class UnitCoreManager {
        protected ArrayList<UnitCore<?>> buffer = new ArrayList<>();
        protected int count = 0;//how many nodes have been borrowed
        
        public final int count() { return count; }
        public final int size() { return buffer.size(); }
        
        //<editor-fold defaultstate="collapsed" desc="running-area">
        public synchronized UnitCore<?> get() {
            if(count >= buffer.size()) buffer.add(create_unit_core());
            return buffer.get(count++);//point to the next unit_core 
        }
        public synchronized UnitCore<?> head() { return buffer.get(0); }
        public synchronized UnitCore<?> get(int index) {
            if(index >= count) throw new IllegalArgumentException("index out of range");
            else if(index < 0) index += count;
            return buffer.get(index);
        }
        
        public void variables(TensorSet set) {
            for(int i=0; i<count;i++) buffer.get(i).variables(set); 
        }
        
        public void gc() { 
            for (int i=0; i<count; i++) buffer.get(i).gc(); 
            count = 0;
        }
        
        public synchronized  void clear() { 
            buffer.forEach((u) -> { u.gc(); });
            buffer.clear();
            count = 0; 
        }
        //</editor-fold>
    }
    //</editor-fold>
    protected UnitCoreManager create_ucm() { return new UnitCoreManager(); } 
    transient protected UnitCoreManager ucm = create_ucm();
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    protected String name = default_name();
    public String default_name() { return getClass().getSimpleName(); }
    public String name() { return name; }
    public Unit name(String name) { this.name = name; return this;}
    
    transient public static final String add_pre = "  ";
    public <T extends Unit> T println() { System.out.println(toString()); return (T) this; }
    public abstract void append(String pre, StringBuilder sb);
    public final void append(StringBuilder sb) { append("", sb); }
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(64);
        this.append(sb);
        return sb.toString();
    }
    
    public <T extends UnitCore<?>> ArrayList<T> unit_cores() { return (ArrayList<T>) ucm.buffer; }

    @Override
    protected void finalize() throws Throwable {
        super.finalize(); 
        this.gc();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="functions: find">
    public void find(UnitSet set, Class<? extends Unit> cls) {
        if(cls.isAssignableFrom(getClass())) set.add(this);
    }
    
    public <T extends Unit> Set<T> find() {
        UnitSet set = new UnitSet();
        find(set, Unit.class);
        return (Set<T>) set; 
    }
    
    public <T extends Unit> Set<T> find(Class<T> cls) { 
        UnitSet set = new UnitSet();
        find(set, cls);
        return (Set<T>) set; 
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="functions: variables & params & state">  
    public void variables(TensorSet set) { ucm.variables(set); } 
    public Set<Tensor> variables() { 
        TensorSet set = new TensorSet(4);
        variables(set); 
        return set; 
    }
   
    public abstract void params(ParamSet set);
    public Set<Parameter> params() { 
        ParamSet set = new ParamSet(4); 
        params(set); 
        return set; 
    }
    
    public abstract void param_map(ParamMap<String> map);
    public Map<String, Parameter> param_map() { 
        ParamMap<String> map = new ParamMap<>(4); 
        param_map(map); 
        return map; 
    }
    
    @Override public State read() { return state(); }
    @Override public abstract void update_state(State dic, boolean partial);
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    public abstract int input_tensor_num();
    public abstract int output_tensor_num();
    public abstract boolean is_complex();
    
    transient private boolean backward_grads = true;
    @Override public boolean backward_grads() { return backward_grads;}
    @Override public Unit backward_grads(boolean flag) { backward_grads = flag; return this; }
    
    public boolean need_grads() {
        boolean flag = false;
        for(Parameter param : params()) flag = (flag || param.need_grads());
        return flag;
    }
    public Unit need_grads(boolean flag) {
        for(Parameter param : params()) param.need_grads(flag);
        return this;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: working"> 
    public int index_of_core(UnitCore core) { return ucm.buffer.indexOf(core); }
    protected abstract UnitCore<?> create_unit_core();
    
    protected abstract void __init__(Engine eg);
    public <T extends Unit> T init(Engine eg) { init(eg, name); return (T) this; }
    public <T extends Unit> T init(Engine eg, String name) {
        this.name = name;
        __init__(eg); 
        return (T) this;
    }
    
    public Tensor[] forward(Tensor... X) {
        UnitCore core = ucm.get();
        if (this.before_backward != null) core.hook_before_backward(before_backward);
        if (this.after_backward != null) core.hook_after_backward(after_backward);
        return core.forward(X); 
    }
    
    public Tensor[] backward(int index, Tensor... grad) { return ucm.get(index).backward(grad);  }
    public Tensor[] backward(Tensor... grad) { return ucm.head().backward(grad); }
    
    public synchronized void gc() { ucm.gc(); }
    public synchronized void delete() {
        ucm.clear();//delete all variables, and set references to nullptr
        params().forEach((Parameter t)-> { t.delete(); });
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: hook-mechanism">
    transient protected Hook before_backward;
    transient protected Hook after_backward;
    
    @Override public Unit hook_before_backward(Hook hook) { before_backward = hook; return this; }
    @Override public Unit hook_after_backward(Hook hook) { after_backward = hook; return this; }
    //</editor-fold>
}
