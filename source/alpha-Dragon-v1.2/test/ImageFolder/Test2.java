/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ImageFolder;

import java.io.File;
import java.util.ArrayList;
import z.util.lang.SimpleTimer;

/**
 *
 * @author Gilgamesh
 */
public class Test2
{
    public static void test1()
    {
        SimpleTimer timer = SimpleTimer.clock();
        
        int dir_count = 0, img_count = 0;
        
        ArrayList<String> imgs = new ArrayList<>(1 << 20);
        
        File dirs = new File("E:\\data\\ImageNet2012\\ILSVRC2012\\ILSVRC2012_img_train");
        for(File dir : dirs.listFiles())
        {
            System.out.println(dir_count); dir_count++;
            for(String f : dir.list()) { imgs.add(f); img_count++; }
//            if(dir_count == 100) break;
            System.gc();
        }
        
        System.out.println();
        System.out.println("count = " + dir_count);
        System.out.println("num = " + img_count);
        
        long time = timer.record().timeStamp_dif_millis();
        System.out.println("time = " + time);
    }
    
    public static void main(String[] args)
    {
//        System.out.println(Runtime.getRuntime().totalMemory());
//        System.out.println(Runtime.getRuntime().freeMemory());
       
        test1();
    }
}
