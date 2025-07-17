#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_112_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<112> mm_instantiator;

public:
  UnalignedM_112_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_112_MMInstantiatorWrapper__;

}  // namespace LLMMM
