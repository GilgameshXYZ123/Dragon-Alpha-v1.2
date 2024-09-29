/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data;

import java.util.Map;
import z.dragon.data.container.DataContainer;
import java.io.File;
import java.util.Arrays;
import java.util.Comparator;
import java.util.TreeMap;
import z.dragon.data.container.ListContainer;

/**
 *
 * @author Gilgamesh
 */
public class FileFolder extends DataSet<File, Integer> {
    public static Comparator<File> file_comparator = (a, b) -> a.getName().compareTo(b.getName());
    
    protected Map<String, Integer> labels;
    protected File root_dir;
    protected boolean inited = false;
    
    public FileFolder(DataContainer<File, Integer> conta, 
            File root_dir, Map<String, Integer> labels) 
    {
        super(conta);
        root_dir(root_dir);
        this.labels = labels;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public File root_dir() { return root_dir; }
    public FileFolder root_dir(String root_dir) { return root_dir(new File(root_dir)); }
    public FileFolder root_dir(File root_dir) { 
        if(inited) throw new RuntimeException("This FileFolder has inited");
        if(!root_dir.exists()) throw new IllegalArgumentException(String.format("root_dir { %s } does not exist", root_dir));
        if(!root_dir.isDirectory()) throw new IllegalArgumentException(String.format("root_dir { %s } is not a directory", root_dir));
        this.root_dir = root_dir; return this;
    };
    
    public Map<String, Integer> labels() { return labels; }
    public FileFolder labels(Map<String, Integer> labels) {
        if(inited) throw new RuntimeException("This FileFolder has inited");
        this.labels = labels;
        return this; 
    }
    
    public Integer label(String dir_name) { return labels.get(dir_name); }
    public Integer[] label(String... dir_names) {
        Integer[] key = new Integer[dir_names.length];
        for(int i=0; i<dir_names.length; i++) key[i] = labels.get(dir_names[i]);
        return key;
    }
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append(" { ");
        sb.append("size = ").append(size());
        sb.append(", input_class").append(input_class());
        sb.append(", label_class").append(label_class());
        sb.append(", inited = ").append(inited);
        sb.append(", num_class = ").append(labels == null ? 0 : labels.size());
        sb.append(", root_dir = ").append(root_dir);
        sb.append("}");
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override public FileFolder[] split_clear(int sub_size) { FileFolder[] subs = split(sub_size); clear(); return subs; }
    @Override public FileFolder[] split_clear(float percent) { FileFolder[] subs = split(percent); clear(); return subs; }
    @Override public FileFolder[] split(float percent) { return split((int)(size() * percent)); }
    @Override public FileFolder[] split(int sub_size) {
        DataContainer<File, Integer>[] contas = con.split(sub_size);
        FileFolder first = new FileFolder(contas[0], root_dir, labels);
        FileFolder last  = new FileFolder(contas[1], root_dir, labels);
        return new FileFolder[]{ first, last };
    }

    @Override public FileFolder[] class_split_clear(float percent) { FileFolder[] subs = class_split(percent); clear(); return subs; }
    @Override public FileFolder[] class_split(float percent) {
        DataContainer<File, Integer>[] contas = con.class_split(percent);
        FileFolder first = new FileFolder(contas[0], root_dir, labels);
        FileFolder last  = new FileFolder(contas[1], root_dir, labels);
        return new FileFolder[]{ first, last };
    }
    
    public synchronized FileFolder init() {
        if (inited) return this;
       
        File[] dirs = root_dir.listFiles();
        if(labels == null || labels.isEmpty()) {//default method to init label
            Arrays.sort(dirs, file_comparator);
            labels = new TreeMap<>();
            for(int i=0; i<dirs.length; i++) labels.put(dirs[i].getName(), i);
        }
        else if(labels.size() != dirs.length) 
            throw new IllegalArgumentException(String.format(
                    "labels.size { got %d } != numbers of directorys { got %d }",
                    labels.size(), dirs.length));
        
        //======[load data from dirs]===========================================
        File[] files = dirs[0].listFiles();
        if((con instanceof ListContainer))//ensure size for container if attainable
            ((ListContainer) con).ensureCapacity(files.length * dirs.length);
           
        int label = labels.get(dirs[0].getName());
        for(File f : files) con.add(f, label);
        
        for(int i=1; i<dirs.length; i++) {
            label = labels.get(dirs[i].getName());
            for(File f : dirs[i].listFiles()) con.add(f, label);
        }
        
        inited = true;
        return this;
    }
    //</editor-fold>
}
