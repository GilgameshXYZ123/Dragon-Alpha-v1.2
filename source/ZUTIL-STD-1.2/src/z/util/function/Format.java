/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.function;

/**
 * <pre>
 * Format the data form the input-type I to output-type O.
 * (1) to improve the performance of Format, you can implement batch process, ex:
 *      =>let I=int[], O=double[].
 * </pre>
 * @author dell
 * @param <I>
 * @param <O>
 */
public interface Format<I,O>
{
    public O format(I values);
}
