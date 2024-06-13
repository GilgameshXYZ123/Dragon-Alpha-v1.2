/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.xml.exception;

import org.w3c.dom.Node;

/**
 *
 * @author dell
 */
public class NullAttributeException extends NullPointerException
{
    private static final String MSG_F="Node[";
    private static final String MSG_B="].attribute is null";
    
    public NullAttributeException() {}
    public NullAttributeException(String s) 
    {
        super(s);
    }
    public NullAttributeException(Node n)
    {
        super(MSG_F+n.getNodeName()+MSG_B);
    }
}
