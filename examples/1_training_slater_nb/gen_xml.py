import xml.etree.ElementTree as ET
import numpy as np
import jax.numpy as jnp
import os
import pickle
from dmff.api import Hamiltonian
from typing import Dict, Any, Optional

def update_force_field(params_pickle: str, input_xml: str, output_dir: str = 'output_xml') -> str:
    """
    Update force field parameters and generate a new XML file
    
    Args:
        params_pickle: path to the parameters pickle file
        input_xml: path to the input force field XML file
        output_dir: output directory, defaults to 'output_xml'
    
    Returns:
        str: path to the output XML file
    """
    def params_convert(params: Dict) -> Dict:
        # Initialize all force field parameter dictionaries
        force_params = {
            'SlaterExForce': {},
            'SlaterSrEsForce': {},
            'SlaterSrPolForce': {},
            'SlaterSrDispForce': {},
            'SlaterDhfForce': {},
            'QqTtDampingForce': {},
            'SlaterDampingForce': {}
        }
        
        # Set B parameters
        for force in force_params.values():
            force['B'] = params['B']
            # force['C'] = params['C']
        
        # Set A parameters
        force_params['SlaterExForce']['A'] = params['A_ex']
        force_params['SlaterSrEsForce']['A'] = params['A_es']
        force_params['SlaterSrPolForce']['A'] = params['A_pol']
        force_params['SlaterSrDispForce']['A'] = params['A_disp']
        force_params['SlaterDhfForce']['A'] = params['A_dhf']
        
        # Set damping parameters
        force_params['QqTtDampingForce']['Q'] = params['Q']
        force_params['SlaterDampingForce'].update({
            'C6': params['C6'],
            'C8': params['C8'],
            'C10': params['C10']
        })
        
        return force_params

    def get_params(restart: str, params0: Dict) -> Dict:
        """Get parameters"""
        with open(restart, 'rb') as ifile:
            return pickle.load(ifile)

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get initial parameters
        params0 = Hamiltonian(input_xml).getParameters()
        params = get_params(params_pickle, params0)
        force_values = params_convert(params)
        
        # Read and update XML
        tree = ET.parse(input_xml)
        root = tree.getroot()
        
        # Update parameters
        for force_type, values in force_values.items():
            for elem in root.iter(force_type):
                for i, atom_elem in enumerate(elem.iter('Atom')):
                    for param in ['A', 'B']:
                        if param in values:
                            atom_elem.set(param, str(float(values[param][i])))
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(params_pickle))[0]
        output_name = f"output{base_name.split('params')[-1]}.xml"
        # output_name = f"output.xml"
        output_path = os.path.join(output_dir, output_name)
        
        # Save XML
        tree.write(output_path)
        print(f'Saved updated force field file to: {output_path}')
        
        return output_path
        
    except Exception as e:
        print(f"An error occurred while updating the force field file: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    params_pickle = "params/params.pickle"  # Replace with your parameter file path
    input_xml = "phyneo_ecl.xml"  # Replace with your input XML file path
    output_dir = "output"  # Output directory
    
    # Update force field and generate XML
    output_xml_path = update_force_field(params_pickle, input_xml, output_dir)
    
    # Validate the generated XML file
    print(f"\nValidating generated XML file: {output_xml_path}")
    try:
        # Try loading the force field to validate format
        from dmff.api import Hamiltonian
        hamiltonian = Hamiltonian(output_xml_path)
        print("Force field file loaded successfully!")
        
        # Print some parameters for inspection
        print("\nPartial parameter examples:")
        params = hamiltonian.getParameters()
        for force_type in ['SlaterExForce', 'SlaterSrEsForce', 'SlaterSrPolForce', 'SlaterSrDispForce', 'SlaterDhfForce']:
            if force_type in params:
                print(f"{force_type}:")
                for param_name in ['A', 'B']:
                    if param_name in params[force_type]:
                        print(f"  {param_name}: {params[force_type][param_name]}")
                        
    except Exception as e:
        print(f"Failed to validate XML file: {e}")
