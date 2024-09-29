/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.complex;

import z.dragon.nn.core.module.CoreModule;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import z.dragon.common.state.State;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Train2Eval;
import z.dragon.nn.unit.Unit;
import z.dragon.common.state.State.AsyncStateUpdate;
import z.dragon.engine.Parameter;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.core.UnitCore;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@SuppressWarnings("unchecked")
@Passed("CudaFloat32Base")
public abstract class Module extends Unit implements Train2Eval, AsyncStateUpdate {   
    private static final long serialVersionUID = 4124125555122341L;
    private boolean constructed = false;
    private final UnitMap<String>   unit_map  = new UnitMap<>();//<unit, fid_name>
    private final Map<String, Unit> runit_map = new HashMap<>();//<fid_name, unit>
  
    //<editor-fold defaultstate="collapsed" desc="register_member_units">
    private synchronized void register_member_units() {
        if(constructed) return;//constructed only once
        try {
            for(Field fid : getClass().getDeclaredFields()) {
                fid.setAccessible(true);
                Object member = fid.get(this);
                if(member instanceof Unit) {
                    Unit u = (Unit) member; 
                    String fid_name = fid.getName();
                    
                    u.name(name + '.' + fid_name);//recusively set unit.name
                    unit_map.put(u, fid_name);//<unit, fid_name>
                    runit_map.put(fid_name, u);//<fid_name, unit>
                }
            }
        }
        catch(IllegalAccessException | IllegalArgumentException | SecurityException e) {
            throw new RuntimeException(e);
        }
        constructed = true;//change the flag of 'constructed'
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Basic-functions">
    @Override 
    public Module name(String name) {
        register_member_units();
        this.name = name; //the name of tail or head is redundant
        unit_map.forEach((Unit u, String fid_name)->{ u.name(name + '.' + fid_name); });
        return this;
    }
    
    public int size() { register_member_units(); return unit_map.size(); }
    public Set<Unit> units() { register_member_units(); return unit_map.keySet(); }
    public Map<String, Unit> units_map() { register_member_units(); return runit_map; }
    public <T extends Unit> T unit(String fid_name) { return (T) units_map().get(fid_name); }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        register_member_units();
        sb.append(default_name()).append(" { size = ").append(size());
        String next_pre = pre + add_pre;
        try//can't use unit_map, as it may shuffle the order of member params
        {
            for(Field fid : getClass().getDeclaredFields()) {
                fid.setAccessible(true);
                Object member = fid.get(this);
                if(member instanceof Unit) {
                    Unit u = (Unit) member; 
                    String fid_name = fid.getName();
                    
                    sb.append('\n').append(pre);//start a new line
                    sb.append('(').append(fid_name).append(") ");
                    if(u.is_complex()) u.append(next_pre, sb);
                    else u.append(sb);
                }
                fid.setAccessible(false);
            }
        }
        catch(IllegalAccessException | IllegalArgumentException | SecurityException e) {
            throw new RuntimeException(e);
        }
        sb.append('\n').append(pre).append('}');
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(512);
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override public boolean is_complex() { return true; }
    @Override public int input_tensor_num() { return -1; }
    @Override public int output_tensor_num() { return -1; }
    
    @Override 
    protected UnitCore create_unit_core() {  
        return new CoreModule(this); 
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="runnning-area: find">
    @Override
    public void find(UnitSet set, Class<? extends Unit> cls) {
        register_member_units();
        super.find(set, cls);//add this, if this.class is a subclass of clazz
        unit_map.keySet().forEach((u) -> { u.find(set, cls); });
    }
    
    @Override
    public <T extends Unit> Set<T> find(Class<T> cls) {
        UnitSet set = new UnitSet(unit_map.size() << 1);
        find(set, cls);
        return (Set<T>) set;
    }
    
    @Override
    public <T extends Unit> Set<T> find() {
        UnitSet set = new UnitSet(unit_map.size() << 1);
        find(set, Unit.class); 
        return (Set<T>) set;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="functions: params"> 
    @Override 
    public void params(ParamSet set) { 
        register_member_units(); 
        unit_map.keySet().forEach((u) -> { u.params(set); });
    }
    @Override 
    public Set<Parameter> params() {
        ParamSet set = new ParamSet(unit_map.size() << 1); 
        params(set); 
        return set; 
    }
    
    @Override
    public void param_map(ParamMap<String> map) { 
        register_member_units(); 
        unit_map.keySet().forEach((u) -> { u.param_map(map); });
    }
    @Override
    public Map<String, Parameter> param_map() {
        ParamMap<String> map = new ParamMap<>(unit_map.size() << 1); 
        param_map(map); 
        return map; 
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="functions: state">
    @Override 
    public void state(State dic) { 
        register_member_units(); 
        unit_map.keySet().forEach((u) -> { u.state(dic); });
    }
    @Override 
    public State state() { 
        State dic = new State(unit_map.size() << 1); 
        state(dic); 
        return dic;  
    }
    
    //update state async--------------------------------------------------------
    private boolean update_state_sync = false;
    @Override public boolean update_state_sync() { return update_state_sync; }
    @Override 
    public <T extends AsyncStateUpdate> T update_state_sync(boolean flag) { 
        update_state_sync = flag; return (T) this; 
    }
    
    @Override
    public void update_state(State dic, boolean partial, List<Future<?>> fts) {
        register_member_units();
        
        if(fts == null) { //sync update_state mode
            unit_map.keySet().forEach((u) -> {
                if(u instanceof AsyncStateUpdate) 
                    ((AsyncStateUpdate)u).update_state(dic, partial, null);
                else u.update_state(dic, partial);
            });
            return; 
        }
        
        unit_map.keySet().forEach((u) -> {//async update_state mode
            if(u instanceof AsyncStateUpdate)
                ((AsyncStateUpdate)u).update_state(dic, partial, fts);
            else fts.add(exec.submit(()-> { u.update_state(dic, partial); }));
        });
    }
    
    @Override
    public void update_state(State dic, boolean partial) { 
        ArrayList<Future<?>> fts = (update_state_sync? null: new ArrayList<>(unit_map.size()));
        this.update_state(dic, partial, fts);
        
        if(fts == null || fts.isEmpty()) return;
        try { for(Future<?> ft : fts)  ft.get(); }
        catch(InterruptedException | ExecutionException e) {
            throw new RuntimeException(e); 
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="functions: Train2Eval">
    @Override
    public boolean training() {
        register_member_units();
        for(Unit u : unit_map.keySet())
            if(u instanceof Train2Eval) 
                if(!((Train2Eval)u).training()) return false;
        return true;
    }
    
    @Override
    public Module train() {
        register_member_units();
        for(Unit u : unit_map.keySet()) 
            if(u instanceof Train2Eval) ((Train2Eval)u).train();
        return this;
    }

    @Override
    public Module eval() {
        register_member_units();
        for (Unit u : unit_map.keySet()) {
            if(u instanceof Train2Eval) ((Train2Eval)u).eval();
        }
        return this;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: working">
    @Override public void __init__(Engine eg) {}
    @Override public synchronized <T extends Unit> T init(Engine eg, String name)  {
        register_member_units();

        this.name = name;
        unit_map.forEach((Unit u, String fid_name)-> { u.init(eg, name + '.' + fid_name); });
        
        __init__(eg);//customized init
        return (T) this;
    }
    
    @Override
    public void variables(TensorSet set) { 
        register_member_units();
        unit_map.keySet().forEach((u) -> { u.variables(set); });
    }
    @Override 
    public Set<Tensor> variables() { 
        TensorSet set = new TensorSet(unit_map.size() << 1); 
        variables(set); 
        return set; 
    }
    
    @Override 
    public synchronized void gc() {  
        ucm.gc();
        for(Unit u : unit_map.keySet()) u.gc(); 
    }
    
    public abstract Tensor[] __forward__(Tensor... X);
    //</editor-fold>
}