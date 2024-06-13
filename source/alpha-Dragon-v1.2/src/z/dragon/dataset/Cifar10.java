/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.dataset;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import static z.dragon.alpha.Alpha.Datas.data;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.common.DragonFile.BufferedFile;
import static z.dragon.common.DragonFile.fl;
import z.dragon.data.DataSet;
import z.dragon.data.Transform;
import z.dragon.data.container.ListContainer;

/**
 * @author Gilgamesh
 */
public final class Cifar10 
{
    private Cifar10() {}
    
    public static final int num_class = 10;//use one hot coding
    public static final int[] picture_dim = {32, 32, 3};//[32 * 32 * 3]
    
    public static final int test_set_size = 10000;
    public static final String test_set_file = "test_batch.bin";
    
    public static final int train_set_size = 50000;
    public static final String[] train_set_files = { 
        "data_batch_1.bin", 
        "data_batch_2.bin",  
        "data_batch_3.bin", 
        "data_batch_4.bin", 
        "data_batch_5.bin"
    };
    
    //<editor-fold defaultstate="collapsed" desc="inner-code">
    static final Transform<byte[][]> __input_transform(Transform<byte[][]> itf) {
        return (itf == null? data.pixel_to_tensor(picture_dim) : itf);
    }
    
    static final String __directory(String dir) {
        if(dir == null) return alpha.home() + "\\data\\cifar10\\";
        if(!dir.endsWith("\\")) return dir + "\\";
        return dir;
    }
    
    static final void load(ListContainer<byte[], Integer> con, String dir, String name) throws IOException {
        String path = __directory(dir) + name;
        byte[] arr = fl.create(path).to_bytes();
        
        for(int i=0, offset = 0; i<10000; i++) {
            int label = arr[offset++];
            byte[] pixel = new byte[3072]; 
            for(int j=0; j<3072; j+=3) pixel[j] = arr[offset++];
            for(int j=1; j<3072; j+=3) pixel[j] = arr[offset++];
            for(int j=2; j<3072; j+=3) pixel[j] = arr[offset++];
            con.add(pixel, label);
        }
    }
    //</editor-fold>
    
    public static Map<Integer, String> labels() { return labels(null); } 
    public static Map<Integer, String> labels(String dir) {
        HashMap<Integer, String> mt; BufferedFile file;
        try {
            file = fl.create(__directory(dir) + "batches.meta.txt");
            String line; int index = 0; mt = new HashMap<>();
            while((line = file.nextLine()) != null) mt.put(index++, line); 
        }
        catch(IOException e) { throw new RuntimeException(e); }
        return mt;
    }
    
    public static DataSet<byte[], Integer> test() { return test(null, null); }
    public static DataSet<byte[], Integer> test(String dir) { return test(dir, null); }
    public static DataSet<byte[], Integer> test(Transform<byte[][]> input_transform) { return test(null, input_transform); }
    public static DataSet<byte[], Integer> test(String dir, Transform<byte[][]> input_transform) {
        ListContainer<byte[], Integer> conta = data.list(byte[].class, Integer.class, test_set_size);
        try { load(conta, dir, test_set_file); }
        catch(IOException e) { throw new RuntimeException(e); }
        return data.dataset(conta)
                .input_transform(__input_transform(input_transform))
                .label_transform(data.onehot(num_class));
    }
    
    public static DataSet<byte[], Integer> train() { return train(null, null); }
    public static DataSet<byte[], Integer> train(String dir) { return train(dir, null); }
    public static DataSet<byte[], Integer> train(Transform<byte[][]> input_transform) { return train(null, input_transform); }
    public static DataSet<byte[], Integer> train(String dir, Transform<byte[][]> input_transform) {
        ListContainer<byte[], Integer> conta = data.list(byte[].class, Integer.class, train_set_size);
        try { for(String file : train_set_files) load(conta, dir, file); }
        catch(IOException e) { throw new RuntimeException(e); }
        return data.dataset(conta)
                .input_transform(__input_transform(input_transform))
                .label_transform(data.onehot(num_class));
    }
}
