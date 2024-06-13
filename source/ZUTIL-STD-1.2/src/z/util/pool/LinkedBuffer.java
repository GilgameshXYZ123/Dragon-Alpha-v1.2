/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.pool;

import z.util.ds.linear.ZLinkedStack;
import z.util.pool.imp.PoolBuffer;

/**
 *
 * @author dell
 * @param <T>
 */
public class LinkedBuffer<T> extends ZLinkedStack<T> implements PoolBuffer<T>
{
}
