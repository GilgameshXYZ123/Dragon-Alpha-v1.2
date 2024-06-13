/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.factory.imp;

/**
 *
 * @author dell
 * @param <A>
 * @param <B>
 * @param <V>
 */
public interface Factoriable<A, B,V> extends NameSpaceLoadable<A, B>,Producable<V>
{
}