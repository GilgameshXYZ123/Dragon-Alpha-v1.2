/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.xml.exception;

import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;


/**
 *
 * @author dell
 */
public class NullNodeException extends NullPointerException
{
    private static final String MSG="Node[";
    
    public NullNodeException() {}
    public NullNodeException(String s)  {
        super(s);
    }
    public NullNodeException(Node n)
    {
        super(MSG+n.getNodeName()+']');
    }
    public NullNodeException(Node n, String msg)
    {
        super(MSG+n.getNodeName()+']'+msg);
    }
}
