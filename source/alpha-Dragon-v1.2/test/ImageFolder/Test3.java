/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ImageFolder;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import z.util.math.ExRandom;

/**
 *
 * @author Gilgamesh
 */
public class Test3 
{
    static ExRandom exr = new ExRandom();
    
    public static void test1() throws FileNotFoundException, IOException
    {
        File f = new File("C:\\Users\\Gilgamesh\\Desktop\\u.txt");
        FileWriter fr = new FileWriter(f);
        BufferedWriter bufr = new BufferedWriter(fr);
        
        int length = 1300000;
        for(int i=0; i<length; i++) {
            String line = exr.nextInt() + " : " + exr.nextInt() + "\n";
            bufr.write(line);
        }
        bufr.close();
    }
    
    public static void main(String[] args)
    {
        try{
            test1();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        
    }
}
