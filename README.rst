AQUAMI
======
An open source package and GUI for automatic analysis of micrographs containing morphologically complex multiphase materials.  Designed especially for analysis of nanoporous metals and bicontinuous composites. Automatically returns quantitative mean and distribution measurements of microstructural features such as ligament/pore diameter, node-to-node length, particle/pore area, and area fraction.  Measurements and summaries can be conveniently output into excel and pdf files.

For more information, see `this paper <https://doi.org/10.1016/j.commatsci.2017.08.012>`_.

Installing
~~~~~~~~~~

A standalone Windows version that does not require Python can be found `here <https://goo.gl/A8Y9Mq>`_.

The easiest way to install the Python version is with pip.  In the command prompt type:

.. code:: python

    pip install aquami
	
Running the program
~~~~~~~~~~~~~~~~~~~
Type the following in the command prompt pressing the <enter> key between lines:

.. code:: python

	python
	from aquami import gui
	gui.run()
	
New in version 1.1.0
~~~~~~~~~~~~~~~~~~~~

* Fix for Mac operating systems.
* Updates for depreciated functions in numpy and matplotlib.

Requirements
~~~~~~~~~~~~~~~~~~~~
Some functions have been deprecated since I wrote this code several years ago. I've added a requirements.txt file that specifies versions of the dependencies I used when writing the code. (Github is warning me that Pillow 4.0.0 has some vulnerabilities, so you can try the latest package for Pillow.)
