#pragma once
#include "Constants.h"
#include "Ensemble.h"
#include "Grid.h"
#include "AnalyticalField.h"
#include "Pusher.h"
#include "Vectors.h"

#include "compton.h"
#include "breit_wheeler.h"
#include "macros.h"

#include <stdint.h>
#include <omp.h>
#include <random>

///
/// Added the separate randGenerator for every thread
/// 

using namespace constants;
namespace pfc
{
    template <class TGrid>  // may be AnalyticalField or any Grid type
    class Scalar_Fast_QED : public ParticlePusher
    {
    public:

        Scalar_Fast_QED() : compton(), breit_wheeler(),
            schwingerField(sqr(Constants<FP>::electronMass() * Constants<FP>::lightVelocity())
                * Constants<FP>::lightVelocity() / (-Constants<FP>::electronCharge() * Constants<FP>::planck()))
        {
            int maxThreads = OMP_GET_MAX_THREADS();

            this->randGenerator.resize(maxThreads);
            this->distribution.resize(maxThreads);
            for (int thr = 0; thr < maxThreads; thr++) {
                this->randGenerator[thr].seed();
                this->distribution[thr] = std::uniform_real_distribution<FP>(0.0, 1.0);
            }
        }

        void disablePhotonEmission() 
        {
            this->photonEmissionEnabled = false;
        }

        void enablePhotonEmission()
        {
            this->photonEmissionEnabled = true;
        }

        void disablePairProduction()
        {
            this->pairProductionEnabled = false;
        }

        void enablePairProduction()
        {
            this->pairProductionEnabled = true;
        }

        void processParticles(Ensemble3d* particles, const std::vector<std::vector<FP3>>& e,
            const std::vector<std::vector<FP3>>& b, FP timeStep)
        {
            int maxThreads = OMP_GET_MAX_THREADS();

            std::vector<std::vector<Particle3d>> generatedParticles(maxThreads);
            std::vector<std::vector<Particle3d>> generatedPhotons(maxThreads);

            if ((*particles)[Photon].size())
                handlePhotons((*particles)[Photon], e[0], b[0], timeStep, generatedParticles, generatedPhotons);
            if ((*particles)[Electron].size())
                handleParticles((*particles)[Electron], e[1], b[1], timeStep, generatedParticles, generatedPhotons);
            if ((*particles)[Positron].size())
                handleParticles((*particles)[Positron], e[2], b[2], timeStep, generatedParticles, generatedPhotons);

            (*particles)[Photon].clear();

            for (int th = 0; th < maxThreads; th++)
            {
                for (int ind = 0; ind < generatedPhotons[th].size(); ind++)
                {
                    particles->addParticle(generatedPhotons[th][ind]);
                }
                for (int ind = 0; ind < generatedParticles[th].size(); ind++)
                {
                    particles->addParticle(generatedParticles[th][ind]);
                }
            }
        }

        // rewrite 2 borises
        void boris(Particle3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void boris(ParticleProxy3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void handlePhotons(ParticleArray3d& photons,
            const std::vector<FP3>& ve, const std::vector<FP3>& vb, FP timeStep,
            std::vector<std::vector<Particle3d>>& generatedParticles,
            std::vector<std::vector<Particle3d>>& generatedPhotons)
        {
            int maxThreads = OMP_GET_MAX_THREADS();

            std::vector<std::vector<Particle3d>> avalancheParticlesThread(maxThreads);
            std::vector<std::vector<FP>> timeAvalancheParticlesThread(maxThreads);
            std::vector<std::vector<Particle3d>> avalanchePhotonsThread(maxThreads);
            std::vector<std::vector<FP>> timeAvalanchePhotonsThread(maxThreads);

#pragma omp parallel for
            for (int i = 0; i < photons.size(); i++)
            {
                FP3 e = ve[i], b = vb[i];

                int threadId = OMP_GET_THREAD_NUM();

                std::vector<Particle3d>& avalancheParticles = avalancheParticlesThread[threadId];
                std::vector<FP>& timeAvalancheParticles = timeAvalancheParticlesThread[threadId];
                std::vector<Particle3d>& avalanchePhotons = avalanchePhotonsThread[threadId];
                std::vector<FP>& timeAvalanchePhotons = timeAvalanchePhotonsThread[threadId];

                // start avalanche with current photon
                avalanchePhotons.push_back(photons[i]);
                timeAvalanchePhotons.push_back((FP)0.0);

                runAvalanche(e, b, timeStep,
                    avalancheParticles, timeAvalancheParticles,
                    avalanchePhotons, timeAvalanchePhotons);

                // push generated physical photons
                for (int k = 0; k != avalanchePhotons.size(); k++)
                    if (avalanchePhotons[k].getGamma() != (FP)1.0)
                        generatedPhotons[threadId].push_back(avalanchePhotons[k]);

                // push generated particles
                for (int k = 0; k != avalancheParticles.size(); k++)
                    generatedParticles[threadId].push_back(avalancheParticles[k]);

                avalancheParticles.clear();
                timeAvalancheParticles.clear();
                avalanchePhotons.clear();
                timeAvalanchePhotons.clear();
            }
        }

        void handleParticles(ParticleArray3d& particles,
            const std::vector<FP3>& ve, const std::vector<FP3>& vb, FP timeStep,
            std::vector<std::vector<Particle3d>>& generatedParticles,
            std::vector<std::vector<Particle3d>>& generatedPhotons)
        {
            int maxThreads = OMP_GET_MAX_THREADS();

            std::vector<std::vector<Particle3d>> avalancheParticlesThread(maxThreads);
            std::vector<std::vector<FP>> timeAvalancheParticlesThread(maxThreads);
            std::vector<std::vector<Particle3d>> avalanchePhotonsThread(maxThreads);
            std::vector<std::vector<FP>> timeAvalanchePhotonsThread(maxThreads);

#pragma omp parallel for
            for (int i = 0; i < particles.size(); i++)
            {
                FP3 e = ve[i], b = vb[i];

                int threadId = OMP_GET_THREAD_NUM();

                std::vector<Particle3d>& avalancheParticles = avalancheParticlesThread[threadId];
                std::vector<FP>& timeAvalancheParticles = timeAvalancheParticlesThread[threadId];
                std::vector<Particle3d>& avalanchePhotons = avalanchePhotonsThread[threadId];
                std::vector<FP>& timeAvalanchePhotons = timeAvalanchePhotonsThread[threadId];

                // start avalanche with current particle
                avalancheParticles.push_back(particles[i]);
                timeAvalancheParticles.push_back((FP)0.0);

                runAvalanche(e, b, timeStep,
                    avalancheParticles, timeAvalancheParticles,
                    avalanchePhotons, timeAvalanchePhotons);

                // push physical photons
                for (int k = 0; k != avalanchePhotons.size(); k++)
                    if (avalanchePhotons[k].getGamma() != (FP)1.0) // include "removePhotonsBelow"
                        generatedPhotons[threadId].push_back(avalanchePhotons[k]);

                // save new position and momentum of current particle
                particles[i].setMomentum(avalancheParticles[0].getMomentum());
                particles[i].setPosition(avalancheParticles[0].getPosition());
                // push generated particles
                for (int k = 1; k != avalancheParticles.size(); k++)
                    generatedParticles[threadId].push_back(avalancheParticles[k]);

                avalancheParticles.clear();
                timeAvalancheParticles.clear();
                avalanchePhotons.clear();
                timeAvalanchePhotons.clear();
            }
        }
        
        void runAvalanche(const FP3& e, const FP3& b, FP timeStep,
            std::vector<Particle3d>& avalancheParticles, std::vector<FP>& timeAvalancheParticles,
            std::vector<Particle3d>& avalanchePhotons, std::vector<FP>& timeAvalanchePhotons)
        {         
            int countProcessedParticles = 0;
            int countProcessedPhotons = 0;

            while (countProcessedParticles != avalancheParticles.size()
                || countProcessedPhotons != avalanchePhotons.size())
            {
                for (int k = countProcessedParticles; k != avalancheParticles.size(); k++)
                {
                    oneParticleStep(avalancheParticles[k], e, b, timeAvalancheParticles[k], timeStep,
                        avalanchePhotons, timeAvalanchePhotons);
                    countProcessedParticles++;
                }
                for (int k = countProcessedPhotons; k != avalanchePhotons.size(); k++)
                {
                    onePhotonStep(avalanchePhotons[k], e, b, timeAvalanchePhotons[k], timeStep,
                        avalancheParticles, timeAvalancheParticles);
                    countProcessedPhotons++;
                }
            }
        }

        void oneParticleStep(Particle3d& particle, const FP3& e, const FP3& b, FP& time, FP timeStep,
            std::vector<Particle3d>& avalanchePhotons, std::vector<FP>& timeAvalanchePhotons)
        {
            while (time < timeStep)
            {
                FP3 v = particle.getVelocity();

                FP H_eff = sqr(e + ((FP)1 / Constants<FP>::lightVelocity()) * VP(v, b))
                    - sqr(SP(e, v) / Constants<FP>::lightVelocity());
                if (H_eff < 0) H_eff = 0;
                H_eff = sqrt(H_eff);
                FP gamma = particle.getGamma();
                FP chi = gamma * H_eff / this->schwingerField;
                FP rate = (FP)0.0, dt = (FP)2*timeStep;
                if (chi > 0.0 && this->photonEmissionEnabled)
                {
                    rate = compton.rate(chi);
                    dt = getDt(rate, chi, gamma);
                }

                if (dt + time > timeStep)
                {
                    boris(particle, e, b, timeStep - time);
                    time = timeStep;
                }
                else
                {
                    boris(particle, e, b, dt);
                    time += dt;

                    FP3 v = particle.getVelocity();
                    FP H_eff = sqr(e + ((FP)1 / Constants<FP>::lightVelocity()) * VP(v, b))
                        - sqr(SP(e, v) / Constants<FP>::lightVelocity());
                    if (H_eff < 0) H_eff = 0;
                    H_eff = sqrt(H_eff);
                    FP gamma = particle.getGamma();
                    FP chi_new = gamma * H_eff / this->schwingerField;

                    FP delta = photonGenerator((chi + chi_new) / (FP)2.0);
                    
                    Particle3d newPhoton;
                    newPhoton.setType(Photon);
                    newPhoton.setWeight(particle.getWeight());
                    newPhoton.setPosition(particle.getPosition());
                    newPhoton.setMomentum(delta * particle.getMomentum());

                    avalanchePhotons.push_back(newPhoton);
                    timeAvalanchePhotons.push_back(time);
                    particle.setMomentum(((FP)1 - delta) * particle.getMomentum());
                }
            }
        }

        void onePhotonStep(Particle3d& photon, const FP3& e, const FP3& b, FP& time, FP timeStep,
            std::vector<Particle3d>& avalancheParticles, std::vector<FP>& timeAvalancheParticles)
        {
            FP3 k_ = photon.getVelocity();
            k_ = ((FP)1 / k_.norm()) * k_; // normalized wave vector
            FP H_eff = sqrt(sqr(e + VP(k_, b)) - sqr(SP(e, k_)));
            FP gamma = photon.getMomentum().norm()
                / (Constants<FP>::electronMass() * Constants<FP>::lightVelocity());
            FP chi = gamma * H_eff / this->schwingerField;

            FP rate = 0.0, dt = (FP)2 * timeStep;
            if (chi > 0.0 && this->pairProductionEnabled)
            {
                rate = breit_wheeler.rate(chi);
                dt = getDt(rate, chi, gamma);
            }

            if (dt + time > timeStep)
            {
                photon.setPosition(photon.getPosition()
                    + (timeStep - time) * Constants<FP>::lightVelocity() * k_);
                time = timeStep;
            }
            else
            {
                photon.setPosition(photon.getPosition()
                    + dt * Constants<FP>::lightVelocity() * k_);
                time += dt;
                FP delta = pairGenerator(chi);

                Particle3d newParticle;
                newParticle.setType(Electron);
                newParticle.setWeight(photon.getWeight());
                newParticle.setPosition(photon.getPosition());
                newParticle.setMomentum(delta * photon.getMomentum());

                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);

                newParticle.setType(Positron);
                newParticle.setMomentum(((FP)1 - delta) * photon.getMomentum());
                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);

                photon.setP((FP)0.0 * photon.getP());  // mark old photon as deleted
                time = timeStep;
            }
        }
        
        FP getDt(FP rate, FP chi, FP gamma)
        {
            FP r = -log(randomNumberOmp());
            r *= gamma / chi;
            return r / rate;
        }
        
        FP photonGenerator(FP chi)
        {
            FP r = randomNumberOmp();
            return compton.inv_cdf(r, chi);
        }
        
        FP pairGenerator(FP chi)
        {
            FP r = randomNumberOmp();
            if (r < (FP)0.5)
                return breit_wheeler.inv_cdf(r, chi);
            else
                return (FP)1.0 - breit_wheeler.inv_cdf((FP)1.0 - r, chi);
        }

        void operator()(ParticleProxy3d* particle, ValueField field, FP timeStep)
        {}

        void operator()(Particle3d* particle, ValueField field, FP timeStep)
        {
            ParticleProxy3d particleProxy(*particle);
            this->operator()(&particleProxy, field, timeStep);
        }

        Compton compton;
        Breit_wheeler breit_wheeler;

    private:

        FP randomNumberOmp()
        {
            int threadId = OMP_GET_THREAD_NUM();
            return distribution[threadId](randGenerator[threadId]);
        }

        std::vector<std::default_random_engine> randGenerator;
        std::vector<std::uniform_real_distribution<FP>> distribution;

        const FP schwingerField;
        
        bool photonEmissionEnabled = true, pairProductionEnabled = true;
    };

    typedef Scalar_Fast_QED<YeeGrid> Scalar_Fast_QED_Yee;
    typedef Scalar_Fast_QED<PSTDGrid> Scalar_Fast_QED_PSTD;
    typedef Scalar_Fast_QED<PSATDGrid> Scalar_Fast_QED_PSATD;
    typedef Scalar_Fast_QED<AnalyticalField> Scalar_Fast_QED_Analytical;
}
