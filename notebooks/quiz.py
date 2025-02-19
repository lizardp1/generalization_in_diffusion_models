# This class defines basic LinkedList functionality
class LinkedList():
# Define the Node class inside the LinkedList class so it is hidden from the outside world
# A Node in a LinkedList has a next pointer and a data element
    class Node():
        def __init__(self, next_ptr = None, data_el = None):
            self.next_ptr = next_ptr
            self.data_el = data_el
# Linked List code
# When initialized with an array, convert the array to a LinkedList (O(n))
    def __init__(self, start_list = None):
        self.head = None
        last = None # we will need to keep track of the previous iteration's Node to append to it
        if start_list is None: return
        for item in start_list:
            if self.head is None: # if we're on the first iteration
                self.head = LinkedList.Node(None, item) # set the head to be the first item
                last = self.head
            else: # general case
                new_last = LinkedList.Node(None, item) # create new last node
                last.next_ptr = new_last # update old last node to point to new one
                last = new_last # set last node to this one
# Append item to linked list in O(n) time
    def append(self, item):
        if self.head is None: # when length is 0, just set to head
            self.head = LinkedList.Node(None, item)
            return
# iterate over linked list
        current_ptr = self.head
        while current_ptr.next_ptr is not None:
            current_ptr = current_ptr.next_ptr
            assert current_ptr is not None # sanity check
# Now, current_ptr should be on the last element
            current_ptr.next_ptr = LinkedList.Node(None, item)
# Print the linked list (O(n))
    def print(self):
# iterate over linked list
        current_ptr = self.head
        i = 0
        while current_ptr is not None:
            print(f"Element {i}: {current_ptr.data_el}")
            current_ptr = current_ptr.next_ptr
            i+= 1
if __name__ == "__main__":
# If we run this python file directly, start a test suite
    ll = LinkedList()
    ll.append("hi 1")
    ll.print() # should print just the first element
    ll.append("hi 2")
    ll.append("hi 3")
    ll.print() # should print the first three elements
    ll = LinkedList([2,3,5,6])
    ll.append(7)
    ll.print() # should print 2,3,5,6,7
