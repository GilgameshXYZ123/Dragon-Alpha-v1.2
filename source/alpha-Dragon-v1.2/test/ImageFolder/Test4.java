/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ImageFolder;

import java.io.File;
import java.util.ArrayList;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class Test4 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp2(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 128);
    
    public static void test1() {
        File dirs = new File("E:\\virtual-disc-V-data\\imagenet2012-train");
        int count = 0;
        for(File dir : dirs.listFiles()) {
            for(File f : dir.listFiles()){ count++; }
        } 
        System.out.println(count);
    }
    
    public static void test2() {
        ArrayList<File> arr = new ArrayList<>(1 << 20);
        File dirs = new File("E:\\virtual-disc-V-data\\imagenet2012-train");
        for(File dir : dirs.listFiles()) {
            for(File f : dir.listFiles()) arr.add(f);
        } 
        System.out.println(arr.size());
    }
    
    public static void main(String[] args) {
        test2();
    }
    
}
