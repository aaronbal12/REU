# -*- coding: utf-8 -*-
# NOCW
# This file is part of the INDUS project

R"""
The :class:`indus_cavity` determines the probe volume for an INDUS calculation
"""

##############################################################################
# Imports
##############################################################################
from __future__ import print_function, division
import platform
if platform.system() in [ 'Windows', 'Darwin' ]:
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg') # turn off interactive plotting
    import matplotlib.pyplot as plt

from willard_chandler import willard_chandler, density_field, wc_interface
from scipy.signal import argrelextrema # used to find first peak
from optparse import OptionParser # Used to allow commands within command line
from parallel import parallel
from datetime import datetime
import mdtraj as md
import numpy as np
import time
import pandas as pd

##############################################################################
# indus_cavity class
##############################################################################

class indus_cavity: 
    R'''
    Determines placement and dimension of an INDUS cavity
    Currently only suppports box geometries
    '''
    def __init__( self, in_file = None, in_path = None, out_path = None, slab_dimensions = '2.0,2.0,0.3', slab_coords = 'none', slab_type = 'density_peak',
                        num_waters = '-1', solvents = [ 'HOH', 'MET' ], convergence = 'False' ):
        print( '**** CLASS: %s ****' %(self.__class__.__name__) )
        if in_file is None:
            raise RuntimeError( '\n    Must input name of .gro and .xtc files. Name must be the same.' )
            
        if in_path is None:
            raise RuntimeError( '\n    Input path not included. Please specify path to trajectory files' )
            
        if out_path is None:
            raise RuntimeError( '\n    Output path not included. Please specify path to output results' )        
        
        self.check_path( in_path )
        self.check_path( out_path )
        
        self.in_file = in_file
        self.in_path = in_path
        self.out_path = out_path
        self.slab_dimensions = slab_dimensions
        self.slab_coords = slab_coords
        self.slab_type = slab_type
        self.num_waters = num_waters
        self.solvents = solvents
        self.convergence = convergence
        
    def compute( self ):
        R'''
        Run functions
        '''        
        ## LOAD MD TRAJECTORY USING MDTRAJ        
        traj = self.load_md_trajectory( self.in_path, self.in_file )
            
        if self.convergence == 'False':
            ## COMPUTE INDUS VOLUME PROPERTIES
            num_frames = len(traj)
            data = self.slab_properties( traj[4*num_frames//5:], solvents = self.solvents, slab_dimensions = self.slab_dimensions, slab_coords = self.slab_coords, num_waters = self.num_waters, slab_type = self.slab_type )
        else:
            data = self.check_convergence( traj )

        return traj, data
            
    @staticmethod
    def check_path( path ):
        R'''Checks that path/file exists'''
        import os
        
        ## CHECK
        if not os.path.exists( path ):
            raise RuntimeError( '\n    %s does not exist. Check your inputs' %( path ) )

    def load_md_trajectory( self, path, filename ):
        R'''FUNCTION TO LOAD MD TRAJECTORY USING MDTRAJ'''
        ## CHECKING PATHS
        if path[-1] == '/':
            gro = path + filename + '.gro'
            xtc = path + filename + '.xtc'
        else:
            gro = path + '/' + filename + '.gro'
            xtc = path + '/' + filename + '.xtc'
        
        self.check_path( gro )
        self.check_path( xtc )
        
        ## PRINT LOADING TRAJECTORY
        print( '\n    Loading trajectory from: %s' %( path ) )
        print( '    XTC File: %s' %( xtc ) )
        print( '    GRO File: %s' %( gro ) )
        
        ## LOAD TRAJECTORY
        start = time.time()
        traj = md.load( xtc, top = gro )
        print( '--- Total time for MD load is %s seconds ---' %( time.time() - start ) )
        
        return traj
        
    @staticmethod
    def slab_properties( traj, solvents, slab_dimensions, slab_coords, num_waters, slab_type ):
        R'''Determines the properties of an INDUS slab

        '''
        slab_dimensions = [ float(ii) for ii in slab_dimensions.split(',') ]
        if slab_coords != 'none':
            slab_coords = [ float(ii) for ii in slab_coords.split(',') ]
        num_waters = int(num_waters)
        # if number of waters has an input it takes priorty over sam dimensions
        if num_waters > 0:
            x_dim = slab_dimensions[0]
            y_dim = slab_dimensions[1]
            slab_dimensions = []
            
        # input slab dimensions
        if slab_dimensions:
            x_dim = slab_dimensions[0]
            y_dim = slab_dimensions[1]
            z_dim = slab_dimensions[2]
        
        # get simulation box size
        box_vectors = traj.unitcell_lengths # np.array, nFrames x 3
        
        n_frames = traj.time.size
        box_height = np.mean( box_vectors[:,2] )
        ndx_water = np.array( [ [ atom.index for atom in residue.atoms if 'O' in atom.name ] \
                    for residue in traj.topology.residues if residue.name in [ 'HOH' ]  ] ).flatten() # heavy water atoms
        ndx_MOH = np.array( [ [ atom.index for atom in residue.atoms if 'O' in atom.name ] \
                    for residue in traj.topology.residues if residue.name in [ 'MOH' ]  ] ).flatten() # heavy MeOH atoms
        ndx_CL = np.array( [ [ atom.index for atom in residue.atoms if 'CL' in atom.name ] \
                    for residue in traj.topology.residues if residue.name in [ 'CL' ]  ] ).flatten() # heavy CL atoms
        
        ## FIND ALL CARBON ATOMS IN LIGANDS (assumes single component systems; for multi-component will only account for tilt)
        monolayer_atoms = [ [ atom.index for atom in residue.atoms if 'C' in atom.name ] for residue in traj.topology.residues
                              if residue.name not in solvents ]
        tail_groups = np.array( [ ligand[-1] for ligand in monolayer_atoms ] )
        monolayer_position = traj.xyz[ :, tail_groups, : ]
        x_mono = np.mean( monolayer_position[:,:,0] )
        y_mono = np.mean( monolayer_position[:,:,1] )
        z_mono = np.mean( monolayer_position[:,:,2] )
#        x_mono = XVAL
#        y_mono = 2.997
        if x_mono < 0.0:
            x_mono += box_vectors[:,0].mean()

        if x_mono > box_vectors[:,0].mean():
            x_mono -= box_vectors[:,0].mean()
        
        if y_mono < 0.0:
            y_mono += box_vectors[:,1].mean()
            
        if y_mono > box_vectors[:,1].mean():
            y_mono += box_vectors[:,1].mean()
            
        # calculate the density profile
        if slab_type == 'density_peak' or slab_type == 'asym_density_peak':
            print( 'Calculating density' )
            bin_width = 0.01
            slab_volume = np.mean( box_vectors[:,0] * box_vectors[:,1] * bin_width )
            
            z_water_coords = traj.xyz[ :, ndx_water, 2 ] # z coords of water molecules
            z_MeOH_coords = traj.xyz[ :, ndx_MOH, 2 ] # z coords of MeOH molecules
            z_CL_coords = traj.xyz[ :, ndx_CL, 2 ] # z coords of MeOH molecules
            # count waters in slabs
            n_bins = np.ceil( box_height / bin_width ).astype('int')
            z = np.arange( bin_width/2., bin_width * ( n_bins + 0.5 ), bin_width )
            
            # 
            # 
            # 
            # 
            density = np.histogram( np.floor( z_water_coords / bin_width ).astype('int'), bins = n_bins, range = ( 0, n_bins ) )[0] / slab_volume / n_frames / (33.34 ) / 0.47 # normalized density
            density_MeOH = np.histogram( np.floor( z_MeOH_coords / bin_width ).astype('int'), bins = n_bins, range = ( 0, n_bins ) )[0] / slab_volume / n_frames / (14.85) / 0.53 # normalized density
            density_CL = np.histogram( np.floor( z_CL_coords / bin_width ).astype('int'), bins = n_bins, range = ( 0, n_bins ) )[0] / slab_volume / n_frames / (0.0543)/100  # normalized density
            
            ndx_peaks = argrelextrema( density, np.greater )
            ndx_peaks_MeOH = argrelextrema( density_MeOH, np.greater )
            ndx_peaks_CL = argrelextrema( density_CL, np.greater )
            ndx_largest2 = np.argsort( density[ndx_peaks] )[-2:] # only look at largest two peaks
            ndx_largest2_MeOH = np.argsort( density_MeOH[ndx_peaks_MeOH] )[-2:] # only look at largest two peaks
            ndx_largest2_CL = np.argsort( density_CL[ndx_peaks_CL] )[-2:] # only look at largest two peaks
            z_peak = z[ndx_peaks][ndx_largest2].min() # choose peak nearest to zero
            z_peak_MeOH = z[ndx_peaks_MeOH][ndx_largest2_MeOH].min() # choose peak nearest to zero
            z_peak_CL = z[ndx_peaks_CL][ndx_largest2_CL].min() # choose peak nearest to zero
            
            if slab_type == 'density_peak':
                slab_position = np.array([ x_mono, y_mono, z_peak ])
                slab_dimensions = np.array([ x_dim, y_dim, z_dim ])
                water_coords = traj.xyz[ :, ndx_water, : ]
                MeOH_coords = traj.xyz[ :, ndx_MOH, : ]
                CL_coords = traj.xyz[ :, ndx_CL, : ]
                in_slab = np.all( np.logical_and( water_coords < slab_position + slab_dimensions/2., 
                                                water_coords > slab_position - slab_dimensions/2. ), axis = 2 )
                in_slab_MeOH = np.all( np.logical_and( MeOH_coords < slab_position + slab_dimensions/2., 
                                                MeOH_coords > slab_position - slab_dimensions/2. ), axis = 2 )
                in_slab_CL = np.all( np.logical_and( CL_coords < slab_position + slab_dimensions/2., 
                                                CL_coords > slab_position - slab_dimensions/2. ), axis = 2 )
                num_waters = in_slab.sum(axis=1).mean()
                num_MeOH = in_slab_MeOH.sum(axis=1).mean()
                num_CL = in_slab_CL.sum(axis=1).mean()
                num_waters = int( num_waters + 10 - ( num_waters % 10 ) )
                num_MeOH = int( num_MeOH + 10 - ( num_MeOH % 10 ) )
                num_CL = int( num_CL + 10 - ( num_CL % 10 ) )
                       
                return { 'slab_position': slab_position,
                         'slab_dimensions': slab_dimensions,
                         'num_waters': num_waters,
                         'num_MeOH' : num_MeOH,
                         'num_CL' : num_CL,
                         'density': np.array([ z, density ]).transpose(),
                         'density_MeOH' : np.array([ z, density_MeOH ]).transpose(),
                         'density_CL' : np.array([ z, density_CL ]).transpose()}       
    
            if slab_type == 'asym_density_peak':
                slab_position = np.array([ x_mono, y_mono, z_peak + 0.25 * z_dim ])
                slab_dimensions = np.array([ x_dim, y_dim, z_dim ])
                water_coords = traj.xyz[ :, ndx_water, : ]
                MeOH_coords = traj.xyz[ :, ndx_MOH, : ]
                CL_coords = traj.xyz[ :, ndx_CL, : ]
                in_slab = np.all( np.logical_and( water_coords < slab_position + slab_dimensions/2., 
                                                water_coords > slab_position - slab_dimensions/2. ), axis = 2 )
                in_slab_MeOH = np.all( np.logical_and( MeOH_coords < slab_position + slab_dimensions/2., 
                                                MeOH_coords > slab_position - slab_dimensions/2. ), axis = 2 )
                in_slab_CL = np.all( np.logical_and( CL_coords < slab_position + slab_dimensions/2., 
                                                CL_coords > slab_position - slab_dimensions/2. ), axis = 2 )
                num_waters = in_slab.sum(axis=1).mean()
                num_MeOH = in_slab_MeOH.sum(axis=1).mean()
                num_CL = in_slab_CL.sum(axis=1).mean()
                num_waters = int( num_waters + 10 - ( num_waters % 10 ) )
                num_MeOH = int( num_MeOH + 10 - ( num_MeOH % 10 ) )
                num_CL = int( num_CL + 10 - ( num_CL % 10 ) )
                       
                return { 'slab_position': slab_position,
                         'slab_dimensions': slab_dimensions,
                         'num_waters': num_waters,
                         'num_MeOH' : num_MeOH,
                         'num_CL' : num_CL,
                         'density': np.array([ z, density ]).transpose(),
                         'density_MeOH' : np.array([ z, density_MeOH ]).transpose(),
                         'density_CL' : np.array([ z, density_CL ]).transpose()} 

        if slab_type == 'wc_interface':
            mid_z = 0.5 * traj.unitcell_lengths[:,2].mean()
            kwargs = { 'alpha': 0.24, 'mesh': 0.1  }
            _, avg_spacing = density_field( traj, 0, alpha = 0.24, mesh = 0.1 )

            start = datetime.now()
            p = parallel( traj, willard_chandler, kwargs )
            time_elapsed = datetime.now() - start
            print( 'Time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed) )
            avg_density_field = p.results
        
            data = wc_interface( avg_density_field, avg_spacing, contour = 16. ) # output columns are ordered z, y, x
            data = data[data[:,0]<mid_z,:]
            data.view('i8,i8,i8').sort( order = [ 'f2', 'f1' ], axis = 0 )
            
            if slab_coords != 'none':
                x_mono = slab_coords[0]
                y_mono = slab_coords[1]
            
            dist_sq_center = ( data[:,2] - x_mono )**2. + ( data[:,1] - y_mono )**2.
            closest_ndx = np.nanargmin( dist_sq_center )            
            slab_position = np.array([ x_mono, y_mono, data[closest_ndx,0] + 0.5 * z_dim ]) # shift z where base of cavity is on the avg contour at x,y on the grid
#            slab_position = np.array([ x_mono, y_mono, data[:,0].mean() + 0.5 * z_dim ]) # shift z where base of cavity is on the avg contour
            slab_dimensions = np.array([ x_dim, y_dim, z_dim ])
            water_coords = traj.xyz[ :, ndx_water, : ]
            MeOH_coords = traj.xyz[ :, ndx_MOH, : ]
            CL_coords = traj.xyz[ :, ndx_CL, : ]
            in_slab = np.all( np.logical_and( water_coords < slab_position + slab_dimensions/2., 
                                            water_coords > slab_position - slab_dimensions/2. ), axis = 2 )
            in_slab_MeOH = np.all( np.logical_and( MeOH_coords < slab_position + slab_dimensions/2., 
                                            MeOH_coords > slab_position - slab_dimensions/2. ), axis = 2 )
            in_slab_CL = np.all( np.logical_and( CL_coords < slab_position + slab_dimensions/2., 
                                            CL_coords > slab_position - slab_dimensions/2. ), axis = 2 )
            num_waters = in_slab.sum(axis=1).mean()
            num_MeOH = in_slab_MeOH.sum(axis=1).mean()
            num_CL = in_slab_CL.sum(axis=1).mean()
            num_waters = int( num_waters + 10 - ( num_waters % 10 ) )
            num_MeOH = int( num_MeOH + 10 - ( num_MeOH % 10 ) )
            num_CL = int( num_CL + 10 - ( num_CL % 10 ) )
                   
            return { 'slab_position': slab_position,
                     'slab_dimensions': slab_dimensions,
                     'num_waters': num_waters,
                     'num_MeOH' : num_MeOH,
                     'num_CL' : num_CL,
                     'density': np.array([ z, density ]).transpose(),
                     'density_MeOH' : np.array([ z, density_MeOH ]).transpose(),
                     'density_CL' : np.array([ z, density_CL ]).transpose(),
                     'wc': data }   
       
        if slab_type == 'None':
            return { 'z': z,
                     'density': density,
                     'density_MeOH' : density_MeOH,
                     'density_CL' : density_CL}
            
##############################################################################
# Execute script
##############################################################################

if __name__ == "__main__":    
    print( '*** computing indus slab position ***' )
    
    # --- Define path variables ---    
    testing = 'False'
    if ( testing == 'True' ):
        args = { 'in_file': 'sam_10ps_whole',
                 'in_path':  r'R:/simulations/polar_sams/robustness_checks/sam_single_12x12_300K_dodecanethiol_tip3p_nvt_CHARMM36_2x2x0.3nm/equil/',
                 'out_path': r'R:/simulations/polar_sams/robustness_checks/sam_single_12x12_300K_dodecanethiol_tip3p_nvt_CHARMM36_2x2x0.3nm/output_files/',
                 'slab_dimensions': '2.0,2.0,0.3',
                 'slab_coords': '2.386,3.887',
                 'slab_type': 'wc_interface',
                 'num_waters': '-1',
                 'convergence': "False" }
    else:  
        # Adding options for command line input (e.g. --maxz, etc.)
        use = 'Usage: %prog [options]'
        parser = OptionParser(usage = use)
        parser.add_option( '-f', '--in', dest = 'infile', action = 'store', type = 'string', help = 'input file', default = '.' )
        parser.add_option( '-w', '--wd', dest = 'inpath', action = 'store', type = 'string', help = 'working directory', default = '.' )
        parser.add_option( '-p', '--op', dest = 'outpath', action = 'store', type = 'string', help = 'output directory', default = '.' )
        parser.add_option( '-s', '--slab', dest = 'slab_position_type', action = 'store', type = 'string', help = 'type of slab position', default = 'sam_surface' )
        parser.add_option( '-d', '--dimension', dest = 'slab_dimensions', action = 'store', type = 'string', help = 'dimensions of slab', default = '20,20,8' )
        parser.add_option( '-C', '--coords', dest = 'slab_coords', action = 'store', type = 'string', help = 'slab coordinates', default = 'none' )
        parser.add_option( '-N', '--num_waters', dest = 'num_waters', action = 'store', type = 'string', help = 'number of waters in slab', default = '-1' )
        parser.add_option( '-c', '--convergence', dest = 'convergence', action = 'store', type = 'string', help = 'check convergence', default = 'False' )

        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        args = { 'in_file': options.infile,
                 'in_path':  options.inpath,
                 'out_path': options.outpath,
                 'slab_dimensions': options.slab_dimensions,
                 'slab_coords': options.slab_coords, 
                 'slab_type': options.slab_position_type,
                 'num_waters': options.num_waters,
                 'convergence': options.convergence }

    out_file = args['in_file'].split('_')[0]
    indus = indus_cavity( **args )
    traj, data = indus.compute()
            
    if args['slab_type'] != "None":
        print( 'INDUS slab position: {:0.3f}, {:0.3f}, {:0.3f}'.format( data['slab_position'][0], data['slab_position'][1], data['slab_position'][2] ) )
        
        outfile = open( args['out_path'] + out_file + '_slab_coordinates.csv', 'w+' )
        outfile.write( '# Slab center coords: {:0.3f},{:0.3f},{:0.3f}\n'.format( data['slab_position'][0], data['slab_position'][1], data['slab_position'][2] ) )
        outfile.close()
        
        print( '--- INDUS slab position written to: {:s}'.format( args['out_path'] + out_file + '_slab_coordinates.csv' ) )
    
        print( 'INDUS slab dimensions: {:0.3f}, {:0.3f}, {:0.3f}'.format( data['slab_dimensions'][0], data['slab_dimensions'][1], data['slab_dimensions'][2] ) )
    
        outfile = open( args['out_path'] + out_file + '_slab_dimensions.csv', 'w+' )
        outfile.write( '# Slab dimensions: {:0.3f},{:0.3f},{:0.3f}\n'.format( data['slab_dimensions'][0], data['slab_dimensions'][1], data['slab_dimensions'][2] ) )
        outfile.close()
        
        print( '--- INDUS slab dimensions written to: {:s}'.format( args['out_path'] + out_file + '_slab_dimensions.csv' ) )
        
        print( 'INDUS number of waters in slab: {:d}'.format( data['num_waters'] ) )
    
        outfile = open( args['out_path'] + out_file + '_num_waters.csv', 'w+' )
        outfile.write( '# Number of waters: {:d}\n'.format( data['num_waters'] ) )
        outfile.close()

        print( 'Spitting the density data') 
        density_data = pd.DataFrame(data['density'][:,0])
        density_data[1] = data['density'][:,1]
        density_data[2] = data['density_MeOH'][:,1]
        density_data[3] = data['density_CL'][:,1]
        density_data.to_csv("density_data.csv", header = False, index=False)
        
        print( '--- INDUS number of waters in slab written to: {:s}'.format( args['out_path'] + out_file + '_num_waters.csv' ) )
    
        if 'density' in data:
            print( '--- density profile written to %s' %(args['out_path'] + 'density_profile.png') )
            z = data['density'][:,0]
            rho = data['density'][:,1]
            slab_center = data['slab_position'][2]
            slab_min = data['slab_position'][2] - 0.5 * data['slab_dimensions'][2]
            slab_max = data['slab_position'][2] + 0.5 * data['slab_dimensions'][2]
            
            z_MeOH = data['density_MeOH'][:,0]
            rho_MeOH = data['density_MeOH'][:,1]
            
            z_CL = data['density_CL'][:,0]
            rho_CL = data['density_CL'][:,1]

            plt.figure()
            plt.plot( z, rho )
            plt.plot( z_MeOH, rho_MeOH )
            plt.plot( z_CL, rho_CL )
            plt.plot( [ slab_center, slab_center ], [ 0, 1.5 ], linestyle = '--', color = 'r' )
            plt.plot( [ slab_min, slab_min ], [ 0, 1.5 ], linestyle = '--', color = 'k' )
            plt.plot( [ slab_max, slab_max ], [ 0, 1.5 ], linestyle = '--', color = 'k' )
            plt.ylim([ -0.1, 1.6 ])
            plt.xlabel( r'z (nm)' )
            plt.ylabel( r'$\rho/\rho_{bulk}$ (nm)' ) 
            plt.tight_layout()
            plt.savefig( args['out_path'] + 'density_profile.png', format='png', dpi=1000, bbox_inches='tight' )
            
        if 'wc' in data:
            print( '--- density profile written to %s' %(args['out_path'] + 'willard_chandler_interface.dat') )
            outfile = open( args['out_path'] + 'willard_chandler_interface.dat', 'w+' )
            outfile.write( '# x y z\n\n' )                      
            for line in data['wc']:
                outfile.write( '{:0.3f},{:0.3f},{:0.3f}\n'.format( line[2], line[1], line[0] ) )

            outfile.close()

            with open( args['out_path'] + 'willard_chandler_interface.dat' ) as raw_data:
                data = raw_data.readlines()
            
            wc_data = np.array( [ [ float(el) for el in line.split(',') ] for line in data[2:] ] )

            # WRITE PDB FILE            
            print( '--- pdb file written to %s' %(args['out_path'] + 'willard_chandler_interface.pdb') )
            pdbfile = open( args['out_path'] + 'willard_chandler_interface.pdb', 'w+' )
            pdbfile.write( 'TITLE     frame t=1.000 in water\n' )
            pdbfile.write( 'REMARK    THIS IS A SIMULATION BOX\n' )
            pdbfile.write( 'CRYST1{:9.3f}{:9.3f}{:9.3f}{:>7s}{:>7s}{:>7s} P 1           1\n'.format( traj.unitcell_lengths[0,0]*10, traj.unitcell_lengths[0,1]*10, traj.unitcell_lengths[0,2]*10, '90.00', '90.00', '90.00' ) )
            pdbfile.write( 'MODEL        1\n' )
            for ndx, coord in enumerate( wc_data ):
                line = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format( \
                        'ATOM', ndx+1, 'C', '', 'SUR', '', 1, '', coord[0]*10, coord[1]*10, coord[2]*10, 1.00, 1.00, '', '' )
                pdbfile.write( line )
                
            pdbfile.write( 'TER\n' )
            pdbfile.write( 'ENDMDL\n' )
            pdbfile.close()
            
