#pragma once

#include "Dimension.h"
#include "Grid.h"
#include "Particle.h"
#include "Species.h"
#include <cmath>

namespace pfc {
    class ParticleBoundaryConditions {
    public:
        template <class T_ParticleArray>////// !!!!!!
        void updateParticlePosition(FP3 minCoords, FP3 maxCoords, T_ParticleArray* particleArray) {};////// !!!!!!
    };

    class PeriodicalParticleBoundaryConditions : ParticleBoundaryConditions {
    public:
        template <class T_ParticleArray>////// !!!!!!
        void updateParticlePosition(FP3 minCoords, FP3 maxCoords, T_ParticleArray* particleArray);////// !!!!!!
    };

    template<class T_ParticleArray>
    inline void PeriodicalParticleBoundaryConditions::updateParticlePosition(FP3 minCoords, FP3 maxCoords,////// !!!!!!
        T_ParticleArray* particleArray)
    {
        typedef typename T_ParticleArray::ParticleProxyType ParticleProxyType;
        typedef typename VectorTypeHelper<Dimension::Three, Real>::Type PositionType;
        for (int i = 0; i < particleArray->size(); i++)
        {
            ParticleProxyType particle = (*particleArray)[i];
            PositionType particlePosition = particle.getPosition();
            PositionType newParticlePosition = particle.getPosition();

            if (particlePosition.x < minCoords.x)
                newParticlePosition.x = maxCoords.x - (minCoords.x - particlePosition.x);
            if (particlePosition.x > maxCoords.x)
                newParticlePosition.x = minCoords.x + (particlePosition.x - maxCoords.x);
            
            if (particlePosition.y < minCoords.y)
                newParticlePosition.y = maxCoords.y - (minCoords.y - particlePosition.y);
            if (particlePosition.y > maxCoords.y)
                newParticlePosition.y = minCoords.y + (particlePosition.y - maxCoords.y);
            
            if (particlePosition.z < minCoords.z)
                newParticlePosition.z = maxCoords.z - (minCoords.z - particlePosition.z);
            if (particlePosition.z > maxCoords.z)
                newParticlePosition.z = minCoords.z + (particlePosition.z - maxCoords.z);

            particle.setPosition(newParticlePosition);
        }
    }
}
