#pragma once

#include <random>
#include "Dimension.h"
#include "Particle.h"
#include "VectorsProxy.h"


namespace pfc {
    class ParticleGenerator {
        typedef typename VectorTypeHelper<Three, Real>::Type PositionType;
        typedef typename VectorTypeHelper<Three, Real>::Type MomentumType;
        typedef ParticleTypes TypeIndexType;
        typedef Real WeightType;
    public:

        ParticleGenerator():
            genRound(0), genPosition(0), genMomentum(0) {
            // TODO: seed initialization according to the current domain
        }

        template<class T_ParticleArray>  // !!! not grid type, but step and domain limits
        void operator()(T_ParticleArray* particleArray,
            const Int3& numCells, const FP3& minCoords, const FP3& gridStep,
            std::function<FP(FP, FP, FP)> particleDensity,////// !!!!!!
            std::function<FP(FP, FP, FP)> initialTemperature,////// !!!!!!
            std::function<FP3(FP, FP, FP)> initialMomentum,////// !!!!!!
            WeightType weight = 1.0,
            ParticleTypes typeIndex = ParticleTypes::Electron)
        {
            for (int i = 0; i < numCells.x; ++i)
                for (int j = 0; j < numCells.y; ++j)
                    for (int k = 0; k < numCells.z; ++k) {
                        Int3 minIdx(i, j, k); Int3 maxIdx(i + 1, j + 1, k + 1);
                        FP3 minCellCoords = minCoords + minIdx * gridStep;
                        FP3 maxCellCoords = minCoords + maxIdx * gridStep;

                        this->operator()(particleArray,
                            minCellCoords, maxCellCoords,
                            particleDensity,
                            initialTemperature,
                            initialMomentum,
                            weight, typeIndex // !!!! type index
                            );
                    }
        }

        template<class T_ParticleArray>
        void operator()(T_ParticleArray* particleArray,
            const FP3& minCellCoords, const FP3& maxCellCoords,
            std::function<FP(FP, FP, FP)> particleDensity,////// !!!!!!
            std::function<FP(FP, FP, FP)> initialTemperature,////// !!!!!!
            std::function<FP3(FP, FP, FP)> initialMomentum,////// !!!!!!
            WeightType weight = 1.0,
            ParticleTypes typeIndex = ParticleTypes::Electron)
        {
            FP3 center = (minCellCoords + maxCellCoords) * 0.5;
            // number of particle in cell according to density
            FP expectedParticleNum = particleDensity(center.x, center.y, center.z) *
                (maxCellCoords - minCellCoords).volume() / weight;

            int particleNum = int(expectedParticleNum);
            // random shift of expectedParticleNum by 1 to eliminate the effect of rounding
            std::uniform_real_distribution<FP> distRound(0.0, 1.0);
            if (distRound(genRound) < expectedParticleNum - (FP)particleNum)
                ++particleNum;

            for (int i = 0; i < particleNum; ++i) {
                PositionType particlePosition(getParticleRandomPosition(minCellCoords, maxCellCoords));

                MomentumType meanMomentum = initialMomentum(particlePosition.x, particlePosition.y, particlePosition.z);
                FP temperature = initialTemperature(particlePosition.x, particlePosition.y, particlePosition.z);
                FP mass = ParticleInfo::types[(int)typeIndex].mass;  // !!! mass of a corresponding particle ////// !!!!!!
                MomentumType particleMomentum(getParticleRandomMomentum(meanMomentum, temperature, mass));

                Particle3d newParticle(particlePosition, particleMomentum, weight, typeIndex);  /// !!!!! type index
                particleArray->pushBack(newParticle);
            }
        }

    private:
        
        default_random_engine genRound, genPosition, genMomentum;

        FP3 getParticleRandomPosition(const FP3& minCoord, const FP3& maxCoord) {
            std::uniform_real_distribution<FP> distPosX(minCoord.x, maxCoord.x);
            std::uniform_real_distribution<FP> distPosY(minCoord.y, maxCoord.y);
            std::uniform_real_distribution<FP> distPosZ(minCoord.z, maxCoord.z);
            FP3 position(distPosX(genPosition), distPosY(genPosition), distPosZ(genPosition));
            return position;
        }

        MomentumType getParticleRandomMomentum(const MomentumType& meanMomentum, FP temperature, FP mass) {
            FP alpha = temperature / (mass * constants::c * constants::c) + 1.0;
            alpha = 1.5 / (alpha * alpha - 1.0);
            // !!!!!!!! получаются нули и бесконечности, когда temperature = 0 => sigma=0
            FP sigma = std::sqrt(0.5 / alpha) * mass * constants::c;
            std::normal_distribution<FP> distMom(0.0, 1.0);
            MomentumType momentum(
                meanMomentum.x + sigma * distMom(genMomentum),
                meanMomentum.y + sigma * distMom(genMomentum),
                meanMomentum.z + sigma * distMom(genMomentum)
            );
            return momentum;
        }
    };
}
