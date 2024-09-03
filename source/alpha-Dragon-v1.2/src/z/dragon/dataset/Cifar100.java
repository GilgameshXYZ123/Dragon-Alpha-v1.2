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
 *
 * @author Gilgamesh
 */
public final class Cifar100 {
    private Cifar100() {}
    
    public static final int num_class_fine = 100;
    public static final int[] picture_dim = {32, 32, 3};
    
    public static final String test_file = "test.bin";
    public static int test_set_size = 10000;
    
    public static final String train_file = "train.bin";
    public static int train_set_size = 50000;
    
    //<editor-fold defaultstate="collapsed" desc="inner-code">
    static final Transform<byte[][]> __input_transform(Transform<byte[][]> itf) {
        return (itf == null? data.pixel_to_tensor(picture_dim) : itf);
    }
    
    static final String __directory(String dir) {
        if(dir == null) return alpha.home() + "\\data\\cifar100\\";
        if(!dir.endsWith("\\")) return dir + "\\";
        return dir;
    }
    
    static final void load_fine(ListContainer<byte[], Integer> ds, String dir, String name, int length) throws IOException {
        String path = __directory(dir) + name;
        byte[] arr = fl.create(path).to_bytes();
        
        for(int i=0, offset = 0; i<length; i++) {
            int label = arr[offset]; offset += 2;
            byte[] pixel = new byte[3072];
            for(int j=0; j<3072; j+=3) pixel[j] = arr[offset++];
            for(int j=1; j<3072; j+=3) pixel[j] = arr[offset++];
            for(int j=2; j<3072; j+=3) pixel[j] = arr[offset++];
            ds.add(pixel, label);
        }
    }
    //</editor-fold>
    
    public static Map<Integer, String> fine_labels() { return fine_labels(null); } 
    public static Map<Integer, String> fine_labels(String dir) {
        HashMap<Integer, String> mt; BufferedFile file;
        try {
            file = fl.create(__directory(dir) + "fine_label_names.txt");
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
        try { load_fine(conta, dir, test_file, test_set_size); } 
        catch(IOException e) { throw new RuntimeException(e); }
        return data.dataset(conta)
                .input_transform(__input_transform(input_transform))
                .label_transform(data.onehot(num_class_fine));
    }
    
    public static DataSet<byte[], Integer> train() { return train(null, null); }
    public static DataSet<byte[], Integer> train(String dir) { return train(dir, null); }
    public static DataSet<byte[], Integer> train(Transform<byte[][]> input_transform) { return train(null, input_transform); }
    public static DataSet<byte[], Integer> train(String dir, Transform<byte[][]> input_transform) {
        ListContainer<byte[], Integer> conta = data.list(byte[].class, Integer.class, train_set_size);
        try { load_fine(conta, dir, train_file, train_set_size); }
        catch(IOException e) { throw new RuntimeException(e); }
        if(input_transform == null) input_transform = data.pixel_to_tensor(picture_dim);
        return data.dataset(conta)
                .input_transform(__input_transform(input_transform))
                .label_transform(data.onehot(num_class_fine));
    }
}
